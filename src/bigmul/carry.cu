// carry.cu — carry normalization for the NTT convolution output.
//
// Adapted from crazynds/MillerRabin-GPU (src/ops/carry/carry_norm.cu),
// which offers 4 selectable carry-propagation strategies for batches of
// candidates. Here there is only ever one "candidate" (one multiplication),
// so the batch dimension has been dropped.
//
// Each thread/work-item owns one final 32-bit result limb directly (not a
// 16-bit digit): it reads the two raw base-2^16 convolution coefficients
// that make up that limb, folds them together with an internal carry step,
// and only the carry escaping the *whole limb* is what ripples to the
// neighboring thread/tile. This removes the separate digit-array + final
// packing pass entirely.
//
// PREFIX_SCAN note: it assumes every raw coefficient's carry-out only needs
// binary (0/1) generate/propagate algebra across at most 4 stacked 16-bit
// "planes" (as in the original schoolbook-style accumulator). Our NTT
// convolution coefficients can in principle carry an arbitrarily large
// magnitude (up to ~m * 65535^2), so PREFIX_SCAN is only exact when that
// magnitude does not overflow the 4-plane assumption; SEQUENTIAL,
// SINGLE_TILE and MULTI_TILE handle arbitrary magnitudes correctly.

#include "bigmul/bigmul.cuh"
#include "bigmul/carry.cuh"

static constexpr int LIMB_BITS = 16;
static constexpr uint64_t LIMB_MASK = 0xFFFFu;

// Folds two base-2^16 coefficients (a = low digit's raw value, b = high
// digit's raw value) plus an incoming carry into one packed 32-bit result
// limb, returning the carry that escapes the limb.
__device__ static inline uint32_t combine2(uint64_t a, uint64_t b, uint64_t cin, uint64_t* escape) {
  uint64_t v0 = a + cin;
  uint64_t lo = v0 & LIMB_MASK;
  uint64_t c = v0 >> LIMB_BITS;
  uint64_t v1 = b + c;
  uint64_t hi = v1 & LIMB_MASK;
  *escape = v1 >> LIMB_BITS;
  return (uint32_t)(lo | (hi << 16));
}

// Adds an external carry into an already-packed limb, in place, returning
// whatever escapes past the limb's high digit.
__device__ static inline uint64_t add_carry_into_limb(uint32_t* dst, int limb, uint64_t c) {
  uint32_t w = dst[limb];
  uint64_t escape;
  dst[limb] = combine2(w & LIMB_MASK, (w >> 16) & LIMB_MASK, c, &escape);
  return escape;
}

// ── CARRY_ALG_SEQUENTIAL ─────────────────────────────────────────────────────
#if CARRY_NORM_ALG == CARRY_ALG_SEQUENTIAL

__global__ static void carry_sequential(const uint64_t* __restrict__ d_src,
                                        uint32_t* __restrict__ d_dst, int n_limbs, int n_src) {
  uint64_t carry = 0;
  for (int limb = 0; limb < n_limbs; limb++) {
    int j0 = 2 * limb, j1 = j0 + 1;
    uint64_t a = (j0 < n_src) ? d_src[j0] : 0ULL;
    uint64_t b = (j1 < n_src) ? d_src[j1] : 0ULL;
    d_dst[limb] = combine2(a, b, carry, &carry);
  }
}

// ── CARRY_ALG_SINGLE_TILE ────────────────────────────────────────────────────
// 1 block, CARRY_TILE threads, each owning one 32-bit limb. Shared-memory
// carry propagation across the whole tile per do-while iteration, tile
// carry-out fed into the next tile.
#elif CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE

static_assert(CARRY_TILE >= 32 && (CARRY_TILE % 32) == 0,
              "CARRY_ALG_SINGLE_TILE requires CARRY_TILE to be a multiple of 32");

__global__ static void carry_32bits(const uint64_t* __restrict__ d_src,
                                    uint32_t* __restrict__ d_dst, int n_limbs, int n_src) {
  int tid = threadIdx.x;

  __shared__ uint64_t s_carry[CARRY_TILE];
  __shared__ int s_has_carry[2];
  int hc_idx = 0;

  uint64_t tile_carry = 0;
  int n_tiles = (n_limbs + CARRY_TILE - 1) / CARRY_TILE;

  // Every thread must run the same number of tile/do-while iterations
  // (all __syncthreads below must be reached uniformly), so iterate by
  // tile index rather than an early-exiting stride loop.
  for (int ti = 0; ti < n_tiles; ti++) {
    int limb = ti * CARRY_TILE + tid;
    int j0 = 2 * limb, j1 = j0 + 1;
    uint64_t raw0 = (j0 < n_src) ? d_src[j0] : 0ULL;
    uint64_t raw1 = (j1 < n_src) ? d_src[j1] : 0ULL;

    uint64_t c = (tid == 0) ? tile_carry : 0ULL;
    uint32_t packed = 0;
    uint64_t escape_total = 0;
    bool first = true;

    do {
      hc_idx ^= 1;
      uint64_t a, b;
      if (first) {
        a = raw0;
        b = raw1;
        first = false;
      } else {
        a = packed & LIMB_MASK;
        b = (packed >> 16) & LIMB_MASK;
      }
      uint64_t round_escape;
      packed = combine2(a, b, c, &round_escape);

      s_carry[tid] = round_escape;
      if (tid == 0) s_has_carry[hc_idx] = 0;
      __syncthreads();

      escape_total += s_carry[CARRY_TILE - 1];
      c = (tid > 0) ? s_carry[tid - 1] : 0ULL;

      if (c > 0) s_has_carry[hc_idx] = 1;
      __syncthreads();

    } while (s_has_carry[hc_idx]);

    tile_carry = escape_total;
    if (limb < n_limbs) d_dst[limb] = packed;
  }
}

// ── CARRY_ALG_MULTI_TILE ─────────────────────────────────────────────────────
// Phase 1: parallel intra-tile normalize (limb granularity), exports each
// tile's escape carry. Phase 2: parallel single-hop propagation between
// adjacent tiles. Phase 3: sequential cleanup of any residual multi-tile
// cascade (rare).
#elif CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE

__global__ static void carry_intra_copy(const uint64_t* __restrict__ d_src,
                                        uint32_t* __restrict__ d_dst,
                                        uint64_t* __restrict__ d_tile_carry,
                                        int* __restrict__ d_first_tile, int n_limbs, int n_src) {
  int tile = blockIdx.x, tid = threadIdx.x;
  int n_tiles = (n_limbs + CARRY_TILE - 1) / CARRY_TILE;
  if (tile == 0 && tid == 0) *d_first_tile = n_tiles;

  int limb = tile * CARRY_TILE + tid;
  int j0 = 2 * limb, j1 = j0 + 1;
  uint64_t raw0 = (j0 < n_src) ? d_src[j0] : 0ULL;
  uint64_t raw1 = (j1 < n_src) ? d_src[j1] : 0ULL;

  __shared__ uint64_t s_carry[CARRY_TILE];
  __shared__ int s_has_carry[2];
  int hc_idx = 0;

  uint64_t c = 0;
  uint32_t packed = 0;
  uint64_t escape_total = 0;
  bool first = true;

  do {
    hc_idx ^= 1;
    uint64_t a, b;
    if (first) {
      a = raw0;
      b = raw1;
      first = false;
    } else {
      a = packed & LIMB_MASK;
      b = (packed >> 16) & LIMB_MASK;
    }
    uint64_t round_escape;
    packed = combine2(a, b, c, &round_escape);

    s_carry[tid] = round_escape;
    if (tid == 0) s_has_carry[hc_idx] = 0;
    __syncthreads();

    escape_total += s_carry[CARRY_TILE - 1];
    c = (tid > 0) ? s_carry[tid - 1] : 0ULL;

    if (c > 0) s_has_carry[hc_idx] = 1;
    __syncthreads();

  } while (s_has_carry[hc_idx]);

  if (limb < n_limbs) d_dst[limb] = packed;
  if (tid == 0) d_tile_carry[tile] = escape_total;
}

__global__ static void carry_propagate_tiles(uint32_t* __restrict__ d_dst,
                                             uint64_t* __restrict__ d_tile_carry,
                                             int* __restrict__ d_first_tile, int n_limbs) {
  int t = blockIdx.x * blockDim.x + threadIdx.x + 1;  // receiver tile, 1..n_tiles-1
  int n_tiles = (n_limbs + CARRY_TILE - 1) / CARRY_TILE;
  if (t >= n_tiles) return;

  uint64_t c = d_tile_carry[t - 1];
  if (c != 0) {
    int j_start = t * CARRY_TILE;
    int j_end = min(j_start + CARRY_TILE, n_limbs);
    for (int limb = j_start; c > 0 && limb < j_end; limb++) c = add_carry_into_limb(d_dst, limb, c);
  }
  d_tile_carry[t - 1] = c;
  if (c != 0) atomicMin(d_first_tile, t + 1);
}

__global__ static void carry_inter_tiles(uint32_t* __restrict__ d_dst,
                                         const uint64_t* __restrict__ d_tile_carry,
                                         const int* __restrict__ d_first_tile, int n_limbs) {
  int n_tiles = (n_limbs + CARRY_TILE - 1) / CARRY_TILE;
  int m_start = *d_first_tile;
  if (m_start >= n_tiles) return;

  uint64_t r = 0;
  for (int m = m_start; m < n_tiles; m++) {
    uint64_t c = r + d_tile_carry[m - 2];
    r = 0;
    if (c == 0) continue;
    int j_start = m * CARRY_TILE;
    int j_end = min(j_start + CARRY_TILE, n_limbs);
    for (int limb = j_start; c > 0 && limb < j_end; limb++) c = add_carry_into_limb(d_dst, limb, c);
    r = c;
  }
}

// ── CARRY_ALG_PREFIX_SCAN ────────────────────────────────────────────────────
// Carry-lookahead (Kogge-Stone) normalization, coarsened to 2 base-2^16
// digits (= 1 output limb) per thread. Each thread first locally composes
// its low/high digit's generate/propagate pair into one limb-level (g,p),
// runs the block-wide scan once per channel on that composed pair, then
// re-expands the scanned carry sequentially through its own low then high
// digit. 1 block, CARRY_TILE threads (= CARRY_TILE limbs per tile).
// Only valid when LIMB_BITS == 16 (fixed 4-plane decomposition of the raw
// 64-bit coefficient) — see the file-level note on its magnitude assumption.
#elif CARRY_NORM_ALG == CARRY_ALG_PREFIX_SCAN

static_assert(CARRY_TILE >= 32 && CARRY_TILE <= 1024 && (CARRY_TILE % 32) == 0,
              "CARRY_TILE must be a multiple of 32 between 32 and 1024");

static constexpr unsigned FULL_MASK = 0xFFFFFFFFu;
static constexpr int PSCAN_NWARPS = CARRY_TILE / 32;

struct GP {
  unsigned g, p;
};

__device__ static inline GP gp_of(uint64_t s) {
  GP r;
  r.g = (unsigned)((s >> LIMB_BITS) & 1ULL);
  r.p = ((s & LIMB_MASK) == LIMB_MASK) ? 1u : 0u;
  return r;
}

// Composes two adjacent generate/propagate pairs where `lo` applies first
// (less significant) and `hi` applies second (more significant).
__device__ static inline GP gp_compose(GP lo, GP hi) {
  GP r;
  r.g = hi.g | (hi.p & lo.g);
  r.p = hi.p & lo.p;
  return r;
}

// Block-wide inclusive Kogge-Stone scan of the operator (G,P) over one
// (g,p) per thread. Returns the carry ENTERING this thread's lane (the
// exclusive prefix applied to cin) and writes the block's total carry-out
// to *tile_cout. wG/wP are shared scratch buffers of size PSCAN_NWARPS,
// reusable across sequential calls (each call fully syncs before return).
__device__ static int cla_scan(unsigned g, unsigned p, int cin, unsigned* wG, unsigned* wP,
                               int* tile_cout) {
  int t = threadIdx.x;
  int lane = t & 31;
  int warp = t >> 5;

#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    unsigned gl = __shfl_up_sync(FULL_MASK, g, d);
    unsigned pl = __shfl_up_sync(FULL_MASK, p, d);
    if (lane >= d) {
      g = g | (p & gl);
      p = p & pl;
    }
  }
  unsigned eg = __shfl_up_sync(FULL_MASK, g, 1);
  unsigned ep = __shfl_up_sync(FULL_MASK, p, 1);
  if (lane == 0) {
    eg = 0u;
    ep = 1u;
  }

  if (lane == 31) {
    wG[warp] = g;
    wP[warp] = p;
  }
  __syncthreads();

  unsigned Gpre = 0u, Ppre = 1u;
  unsigned Gtot = 0u, Ptot = 1u;
#pragma unroll
  for (int w = 0; w < PSCAN_NWARPS; w++) {
    unsigned gw = wG[w], pw = wP[w];
    if (w < warp) {
      Gpre = gw | (pw & Gpre);
      Ppre = pw & Ppre;
    }
    Gtot = gw | (pw & Gtot);
    Ptot = pw & Ptot;
  }

  unsigned Cg = eg | (ep & Gpre);
  unsigned Cp = ep & Ppre;
  int carry_in = (int)(Cg | (Cp & (unsigned)cin));

  *tile_cout = (int)(Gtot | (Ptot & (unsigned)cin));

  __syncthreads();
  return carry_in;
}

__global__ static void pscan_normalize(const uint64_t* __restrict__ d_src,
                                       uint32_t* __restrict__ d_dst, int n_limbs, int n_src) {
  int t = threadIdx.x;
  int n_digits = 2 * n_limbs;

  __shared__ uint64_t sraw[2 * CARRY_TILE];
  __shared__ unsigned wG[PSCAN_NWARPS];
  __shared__ unsigned wP[PSCAN_NWARPS];
  __shared__ uint64_t sprev[3];  // raw digit values at positions base-1, base-2, base-3

  if (t < 3) sprev[t] = 0ULL;
  int cinA = 0, cinB = 0, cinC = 0;  // persistent per-channel carry across tile iterations

  for (int base = 0; base < n_digits; base += 2 * CARRY_TILE) {
    int lo_idx = base + 2 * t, hi_idx = lo_idx + 1;
    sraw[2 * t] = (lo_idx < n_src) ? d_src[lo_idx] : 0ULL;
    sraw[2 * t + 1] = (hi_idx < n_src) ? d_src[hi_idx] : 0ULL;
    __syncthreads();

    // raw plane values for a local digit index `k` (0-based within this
    // tile's 2*CARRY_TILE-wide window; may reach back up to 3 into sprev).
    auto raw_at = [&](int k) -> uint64_t {
      if (k >= 0) return sraw[k];
      return sprev[-k - 1];
    };

    int lo = 2 * t, hi = lo + 1;
    uint64_t sA_lo = (raw_at(lo) & LIMB_MASK) + ((raw_at(lo - 1) >> 16) & LIMB_MASK);
    uint64_t sA_hi = (raw_at(hi) & LIMB_MASK) + ((raw_at(hi - 1) >> 16) & LIMB_MASK);
    uint64_t sB_lo = ((raw_at(lo - 2) >> 32) & LIMB_MASK) + ((raw_at(lo - 3) >> 48) & LIMB_MASK);
    uint64_t sB_hi = ((raw_at(hi - 2) >> 32) & LIMB_MASK) + ((raw_at(hi - 3) >> 48) & LIMB_MASK);

    GP gA_lo = gp_of(sA_lo), gA_hi = gp_of(sA_hi);
    GP gA_limb = gp_compose(gA_lo, gA_hi);
    int coutA;
    int cinA_limb = cla_scan(gA_limb.g, gA_limb.p, cinA, wG, wP, &coutA);
    cinA = coutA;
    uint64_t digitA_lo = ((sA_lo & LIMB_MASK) + (uint64_t)cinA_limb) & LIMB_MASK;
    uint64_t midA = gA_lo.g | (gA_lo.p & (unsigned)cinA_limb);
    uint64_t digitA_hi = ((sA_hi & LIMB_MASK) + midA) & LIMB_MASK;

    GP gB_lo = gp_of(sB_lo), gB_hi = gp_of(sB_hi);
    GP gB_limb = gp_compose(gB_lo, gB_hi);
    int coutB;
    int cinB_limb = cla_scan(gB_limb.g, gB_limb.p, cinB, wG, wP, &coutB);
    cinB = coutB;
    uint64_t digitB_lo = ((sB_lo & LIMB_MASK) + (uint64_t)cinB_limb) & LIMB_MASK;
    uint64_t midB = gB_lo.g | (gB_lo.p & (unsigned)cinB_limb);
    uint64_t digitB_hi = ((sB_hi & LIMB_MASK) + midB) & LIMB_MASK;

    uint64_t sC_lo = digitA_lo + digitB_lo;
    uint64_t sC_hi = digitA_hi + digitB_hi;
    GP gC_lo = gp_of(sC_lo), gC_hi = gp_of(sC_hi);
    GP gC_limb = gp_compose(gC_lo, gC_hi);
    int coutC;
    int cinC_limb = cla_scan(gC_limb.g, gC_limb.p, cinC, wG, wP, &coutC);
    cinC = coutC;
    uint64_t digit_lo = ((sC_lo & LIMB_MASK) + (uint64_t)cinC_limb) & LIMB_MASK;
    uint64_t midC = gC_lo.g | (gC_lo.p & (unsigned)cinC_limb);
    uint64_t digit_hi = ((sC_hi & LIMB_MASK) + midC) & LIMB_MASK;

    int limb = base / 2 + t;
    if (limb < n_limbs) d_dst[limb] = (uint32_t)(digit_lo | (digit_hi << 16));

    __syncthreads();
    if (t == 0) {
      sprev[0] = sraw[2 * CARRY_TILE - 1];
      sprev[1] = sraw[2 * CARRY_TILE - 2];
      sprev[2] = sraw[2 * CARRY_TILE - 3];
    }
    __syncthreads();
  }
}

#else
#error \
    "CARRY_NORM_ALG must be CARRY_ALG_SEQUENTIAL, CARRY_ALG_SINGLE_TILE, CARRY_ALG_MULTI_TILE or CARRY_ALG_PREFIX_SCAN"
#endif

// ── entry point ───────────────────────────────────────────────────────────

auto carry_and_assemble(const uint64_t* d_conv, uint32_t* d_result, int n, int m,
                        uint64_t* d_scratch) -> void {
  int n_limbs = 2 * n;

#if CARRY_NORM_ALG == CARRY_ALG_SEQUENTIAL
  carry_sequential<<<1, 1>>>(d_conv, d_result, n_limbs, m);
  check_cuda(cudaGetLastError());

#elif CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE
  carry_32bits<<<1, CARRY_TILE>>>(d_conv, d_result, n_limbs, m);
  check_cuda(cudaGetLastError());

#elif CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
  int n_tiles = (n_limbs + CARRY_TILE - 1) / CARRY_TILE;

  // d_scratch is caller-owned (>= m uint64_t, e.g. an NTT buffer no longer
  // needed at this point) and is far larger than the n_tiles+1 elements
  // needed here, so no allocation happens in this function.
  uint64_t* d_tile_carry = d_scratch;
  int* d_first_tile = reinterpret_cast<int*>(d_scratch + n_tiles);

  carry_intra_copy<<<n_tiles, CARRY_TILE>>>(d_conv, d_result, d_tile_carry, d_first_tile, n_limbs, m);
  check_cuda(cudaGetLastError());
  if (n_tiles > 1) {
    constexpr int THR = 256;
    carry_propagate_tiles<<<(n_tiles - 1 + THR - 1) / THR, THR>>>(d_result, d_tile_carry,
                                                                  d_first_tile, n_limbs);
    check_cuda(cudaGetLastError());
    carry_inter_tiles<<<1, 1>>>(d_result, d_tile_carry, d_first_tile, n_limbs);
    check_cuda(cudaGetLastError());
  }

#elif CARRY_NORM_ALG == CARRY_ALG_PREFIX_SCAN
  pscan_normalize<<<1, CARRY_TILE>>>(d_conv, d_result, n_limbs, m);
  check_cuda(cudaGetLastError());
#endif
}
