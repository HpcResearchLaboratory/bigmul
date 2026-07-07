#include "bigmul/bigmul.cuh"
#include "bigmul/ntt.cuh"

__device__ __inline__ auto mod_mul(uint64_t a, uint64_t b, uint64_t p) -> uint64_t {
  return (uint64_t)(((__uint128_t)a * b) % p);
}

__device__ __inline__ auto mod_add(uint64_t a, uint64_t b, uint64_t p) -> uint64_t {
  uint64_t r = a + b;
  if (r < a || r >= p) r -= p;
  return r;
}

__device__ __inline__ auto mod_sub(uint64_t a, uint64_t b, uint64_t p) -> uint64_t {
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

// Computes a[i] * b[i] mod p on access instead of materializing the product,
// so the pointwise multiply can be fused into the inverse-NTT's bit-reverse
// gather without an extra kernel/buffer round-trip.
struct PointwiseView {
  const uint64_t* a;
  const uint64_t* b;
  uint64_t p;

  __device__ auto operator[](int i) const -> uint64_t { return mod_mul(a[i], b[i], p); }
};

template <typename Src>
__global__ auto bit_reverse_gather(uint64_t* out, Src src, int n, int log_n) -> void {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  int rev = 0, x = i;
  for (int j = 0; j < log_n; j++) {
    rev = (rev << 1) | (x & 1);
    x >>= 1;
  }

  out[i] = src[rev];
}

// When is_last is set, the 1/n normalization (scale) is folded into the same
// write as the last butterfly stage, sparing a separate scale_mod kernel
// pass over the whole array.
__global__ auto butterfly(uint64_t* data, const uint64_t* twiddles, int stage, int n, uint64_t p,
                          bool is_last, uint64_t scale) -> void {
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
  uint64_t r0 = mod_add(u, v, p);
  uint64_t r1 = mod_sub(u, v, p);
  if (is_last) {
    r0 = mod_mul(r0, scale, p);
    r1 = mod_mul(r1, scale, p);
  }
  data[i] = r0;
  data[j] = r1;
}

// n_inv == 0 means "no normalization" (used by the forward transform); since p
// is prime and n is coprime to p, a real 1/n mod p is never 0.
static auto ntt_butterflies(uint64_t* d_data, int n, uint64_t w, uint64_t p, uint64_t n_inv = 0)
    -> void {
  int log_n = __builtin_ctz(n);

  static uint64_t* d_tw = nullptr;
  static size_t tw_pool = 0;
  static uint64_t cached_w = 0, cached_p = 0;
  static int cached_n = 0;
  size_t tw_bytes = n * sizeof(uint64_t);
  if (tw_bytes > tw_pool) {
    if (d_tw) cudaFree(d_tw);
    check_cuda(cudaMalloc(&d_tw, tw_bytes));
    tw_pool = tw_bytes;
  }

  if (w != cached_w || p != cached_p || n > cached_n) {
    constexpr int B = BLOCK_SIZE;
    compute_twiddles<<<(n + B - 1) / B, B>>>(d_tw, w, p, n);
    check_cuda(cudaGetLastError());
    cached_w = w;
    cached_p = p;
    cached_n = n;
  }

  constexpr int B = BLOCK_SIZE;
  for (int stage = 0; stage < log_n; stage++) {
    bool is_last = stage == log_n - 1;
    butterfly<<<(n / 2 + B - 1) / B, B>>>(d_data, d_tw, stage, n, p, is_last && n_inv != 0, n_inv);
    check_cuda(cudaGetLastError());
  }
}

auto ntt_forward(uint64_t* d_data, int n, const NttPrime& prime) -> void {
  int log_n = __builtin_ctz(n);
  constexpr int B = BLOCK_SIZE;
  bit_reverse_permute<<<(n + B - 1) / B, B>>>(d_data, n, log_n);
  check_cuda(cudaGetLastError());

  uint64_t w = mod_pow_host(prime.g, (prime.p - 1) / n, prime.p);
  ntt_butterflies(d_data, n, w, prime.p);
  check_cuda(cudaDeviceSynchronize());
}

auto ntt_inverse(uint64_t* d_data, int n, const NttPrime& prime) -> void {
  int log_n = __builtin_ctz(n);
  constexpr int B = BLOCK_SIZE;
  bit_reverse_permute<<<(n + B - 1) / B, B>>>(d_data, n, log_n);
  check_cuda(cudaGetLastError());

  uint64_t w = mod_pow_host(prime.g, (prime.p - 1) / n, prime.p);
  uint64_t w_inv = mod_pow_host(w, prime.p - 2, prime.p);
  uint64_t n_inv = mod_pow_host((uint64_t)n, prime.p - 2, prime.p);
  ntt_butterflies(d_data, n, w_inv, prime.p, n_inv);
  check_cuda(cudaDeviceSynchronize());
}

// Fuses the pointwise multiply into the inverse-NTT's bit-reverse gather: each
// output slot is filled by computing a[rev(i)] * b[rev(i)] mod p directly via
// PointwiseView::operator[], instead of a separate pointwise_mul kernel
// writing to d_out followed by a pass that reads it back in.
auto ntt_inverse_pointwise(uint64_t* d_out, const uint64_t* d_a, const uint64_t* d_b, int n,
                           const NttPrime& prime) -> void {
  int log_n = __builtin_ctz(n);
  constexpr int B = BLOCK_SIZE;
  PointwiseView src{d_a, d_b, prime.p};
  bit_reverse_gather<<<(n + B - 1) / B, B>>>(d_out, src, n, log_n);
  check_cuda(cudaGetLastError());

  uint64_t w = mod_pow_host(prime.g, (prime.p - 1) / n, prime.p);
  uint64_t w_inv = mod_pow_host(w, prime.p - 2, prime.p);
  uint64_t n_inv = mod_pow_host((uint64_t)n, prime.p - 2, prime.p);
  ntt_butterflies(d_out, n, w_inv, prime.p, n_inv);
  check_cuda(cudaDeviceSynchronize());
}
