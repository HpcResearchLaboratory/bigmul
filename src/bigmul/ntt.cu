#include "bigmul/bigmul.cuh"
#include "bigmul/ntt.cuh"

__device__ auto mod_mul(uint64_t a, uint64_t b, uint64_t p) -> uint64_t {
  return (uint64_t)(((__uint128_t)a * b) % p);
}

__device__ auto mod_add(uint64_t a, uint64_t b, uint64_t p) -> uint64_t {
  uint64_t r = a + b;
  if (r < a || r >= p) r -= p;
  return r;
}

__device__ auto mod_sub(uint64_t a, uint64_t b, uint64_t p) -> uint64_t {
  return a >= b ? a - b : a + p - b;
}

auto mod_pow_host(uint64_t base, uint64_t exp, uint64_t p) -> uint64_t {
  __uint128_t result = 1, b = base % p;
  while (exp > 0) {
    if (exp & 1) result = (result * b) % p;
    b = (b * b) % p;
    exp >>= 1;
  }
  return (uint64_t)result;
}

__device__ auto mod_pow(uint64_t base, uint64_t exp, uint64_t p) -> uint64_t {
  __uint128_t result = 1, b = base % p;
  while (exp > 0) {
    if (exp & 1) result = (result * b) % p;
    b = (b * b) % p;
    exp >>= 1;
  }
  return (uint64_t)result;
}

__global__ auto compute_twiddles(uint64_t* tw, uint64_t w, uint64_t p, int n) -> void {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  tw[i] = mod_pow(w, i, p);
}

__global__ auto bit_reverse_permute(uint64_t* data, int n, int log_n) -> void {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  int rev = 0, x = i;
  for (int j = 0; j < log_n; j++) {
    rev = (rev << 1) | (x & 1);
    x >>= 1;
  }

  if (i < rev) {
    uint64_t tmp = data[i];
    data[i] = data[rev];
    data[rev] = tmp;
  }
}

__global__ auto butterfly(uint64_t* data, const uint64_t* twiddles, int stage, int n, uint64_t p)
    -> void {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n / 2) return;

  int half = 1 << stage;
  int group = k / half;
  int pos = k % half;
  int i = group * 2 * half + pos;
  int j = i + half;
  int step = n >> (stage + 1);

  uint64_t tw = twiddles[step * pos];
  uint64_t u = data[i];
  uint64_t v = mod_mul(data[j], tw, p);
  data[i] = mod_add(u, v, p);
  data[j] = mod_sub(u, v, p);
}

// Fuses `num_stages` consecutive butterfly stages, starting at `start_stage`,
// into a single kernel launch. Butterfly groups at any stage are contiguous
// runs of `2*half` elements, so a block can own one contiguous segment of
// 2^(start_stage+num_stages) elements, load it into shared memory once, run
// all `num_stages` steps there (syncing between them), then write the segment
// back. This avoids the global-memory round trip that `butterfly` pays for
// every single stage. Called repeatedly with increasing `start_stage` to
// cover the whole transform, mirroring several fused launches instead of one
// plain `butterfly` launch per stage.
__global__ auto butterfly_shared(uint64_t* data, const uint64_t* twiddles, int start_stage,
                                 int num_stages, int n, uint64_t p) -> void {
  extern __shared__ uint64_t s[];

  int seg = 1 << (start_stage + num_stages);
  int seg_base = blockIdx.x * seg;
  int tid = threadIdx.x;

  s[tid] = data[seg_base + tid];
  s[tid + seg / 2] = data[seg_base + tid + seg / 2];
  __syncthreads();

  for (int ls = 0; ls < num_stages; ls++) {
    int stage = start_stage + ls;
    int half = 1 << stage;
    int group = tid / half;
    int pos = tid % half;
    int i = group * 2 * half + pos;
    int j = i + half;
    int step = n >> (stage + 1);

    uint64_t tw = twiddles[step * pos];
    uint64_t u = s[i];
    uint64_t v = mod_mul(s[j], tw, p);
    s[i] = mod_add(u, v, p);
    s[j] = mod_sub(u, v, p);
    __syncthreads();
  }

  data[seg_base + tid] = s[tid];
  data[seg_base + tid + seg / 2] = s[tid + seg / 2];
}

__global__ auto scale_mod(uint64_t* data, int n, uint64_t factor, uint64_t p) -> void {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  data[i] = mod_mul(data[i], factor, p);
}

__global__ auto pointwise_mul(uint64_t* out, const uint64_t* a, const uint64_t* b, int n,
                              uint64_t p) -> void {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = mod_mul(a[i], b[i], p);
}

static auto ntt_core(uint64_t* d_data, int n, uint64_t w, uint64_t p) -> void {
  int log_n = __builtin_ctz(n);
  constexpr int B = BLOCK_SIZE;

  static uint64_t* d_tw = nullptr;
  static size_t tw_pool = 0;
  static uint64_t cached_w = 0, cached_p = 0;
  static int cached_n = 0;
  size_t tw_bytes = n * sizeof(uint64_t);
  if (tw_bytes > tw_pool) {
    if (d_tw) cudaFree(d_tw);
    check_cuda(cudaMalloc(&d_tw, tw_bytes));
    tw_pool = tw_bytes;
    if (cached_n != n || cached_w != w || cached_p != p) {
      compute_twiddles<<<(n + B - 1) / B, B>>>(d_tw, w, p, n);
      check_cuda(cudaGetLastError());
      cached_n = n;
      cached_w = w;
      cached_p = p;
    }
  }

  check_cuda(cudaGetLastError());
  bit_reverse_permute<<<(n + B - 1) / B, B>>>(d_data, n, log_n);
  check_cuda(cudaGetLastError());

  // Fuse stages in groups of up to kMaxFusedStages, chaining launches across
  // the whole transform (bounded by the 1024 threads/block limit:
  // seg/2 <= 1024 => seg <= 2048 => 11 stages per group).
  constexpr int kMaxFusedStages = 10;
  int stage = 0;
  while (stage < log_n) {
    int chunk = log_n - stage < kMaxFusedStages ? log_n - stage : kMaxFusedStages;
    int seg = 1 << (stage + chunk);
    size_t shared_bytes = seg * sizeof(uint64_t);
    butterfly_shared<<<n / seg, seg / 2, shared_bytes>>>(d_data, d_tw, stage, chunk, n, p);
    check_cuda(cudaGetLastError());
    stage += chunk;
  }
}

auto ntt_forward(uint64_t* d_data, int n, const NttPrime& prime) -> void {
  uint64_t w = mod_pow_host(prime.g, (prime.p - 1) / n, prime.p);
  ntt_core(d_data, n, w, prime.p);
}

auto ntt_inverse(uint64_t* d_data, int n, const NttPrime& prime) -> void {
  uint64_t w = mod_pow_host(prime.g, (prime.p - 1) / n, prime.p);
  uint64_t w_inv = mod_pow_host(w, prime.p - 2, prime.p);
  ntt_core(d_data, n, w_inv, prime.p);

  uint64_t n_inv = mod_pow_host((uint64_t)n, prime.p - 2, prime.p);
  constexpr int B = BLOCK_SIZE;
  scale_mod<<<(n + B - 1) / B, B>>>(d_data, n, n_inv, prime.p);
  check_cuda(cudaGetLastError());
}

auto ntt_pointwise_mul(uint64_t* d_out, const uint64_t* d_a, const uint64_t* d_b, int n, uint64_t p)
    -> void {
  constexpr int B = BLOCK_SIZE;
  pointwise_mul<<<(n + B - 1) / B, B>>>(d_out, d_a, d_b, n, p);
  check_cuda(cudaGetLastError());
}
