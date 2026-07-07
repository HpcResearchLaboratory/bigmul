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

  static uint64_t* d_tw = nullptr;
  static size_t tw_pool = 0;
  size_t tw_bytes = n * sizeof(uint64_t);
  if (tw_bytes > tw_pool) {
    if (d_tw) cudaFree(d_tw);
    check_cuda(cudaMalloc(&d_tw, tw_bytes));
    tw_pool = tw_bytes;
  }

  constexpr int B = BLOCK_SIZE;
  compute_twiddles<<<(n + B - 1) / B, B>>>(d_tw, w, p, n);
  check_cuda(cudaGetLastError());
  bit_reverse_permute<<<(n + B - 1) / B, B>>>(d_data, n, log_n);
  check_cuda(cudaGetLastError());

  for (int stage = 0; stage < log_n; stage++) {
    butterfly<<<(n / 2 + B - 1) / B, B>>>(d_data, d_tw, stage, n, p);
    check_cuda(cudaGetLastError());
  }

  check_cuda(cudaDeviceSynchronize());
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
  check_cuda(cudaDeviceSynchronize());
}

auto ntt_pointwise_mul(uint64_t* d_out, const uint64_t* d_a, const uint64_t* d_b, int n,
                       uint64_t p) -> void {
  constexpr int B = BLOCK_SIZE;
  pointwise_mul<<<(n + B - 1) / B, B>>>(d_out, d_a, d_b, n, p);
  check_cuda(cudaGetLastError());
  check_cuda(cudaDeviceSynchronize());
}

// ── batched variants ─────────────────────────────────────────────────────
// Each batched kernel keeps the exact per-element math of its single-item
// counterpart, just adding blockIdx.y as the batch index and offsetting
// into that item's n-sized (or n/2-sized) slice. The twiddle table depends
// only on n, so it is computed once and reused across the whole batch.

__global__ auto bit_reverse_permute_batch(uint64_t* data, int n, int log_n, int batch) -> void {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint64_t* d = data + (size_t)blockIdx.y * n;

  int rev = 0, x = i;
  for (int j = 0; j < log_n; j++) {
    rev = (rev << 1) | (x & 1);
    x >>= 1;
  }

  if (i < rev) {
    uint64_t tmp = d[i];
    d[i] = d[rev];
    d[rev] = tmp;
  }
}

__global__ auto butterfly_batch(uint64_t* data, const uint64_t* twiddles, int stage, int n,
                                uint64_t p, int batch) -> void {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n / 2) return;
  uint64_t* d = data + (size_t)blockIdx.y * n;

  int half = 1 << stage;
  int group = k / half;
  int pos = k % half;
  int i = group * 2 * half + pos;
  int j = i + half;
  int step = n >> (stage + 1);

  uint64_t tw = twiddles[step * pos];
  uint64_t u = d[i];
  uint64_t v = mod_mul(d[j], tw, p);
  d[i] = mod_add(u, v, p);
  d[j] = mod_sub(u, v, p);
}

__global__ auto scale_mod_batch(uint64_t* data, int n, uint64_t factor, uint64_t p, int batch)
    -> void {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint64_t* d = data + (size_t)blockIdx.y * n;
  d[i] = mod_mul(d[i], factor, p);
}

__global__ auto pointwise_mul_batch(uint64_t* out, const uint64_t* a, const uint64_t* b, int n,
                                    uint64_t p, int batch) -> void {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  size_t off = (size_t)blockIdx.y * n;
  out[off + i] = mod_mul(a[off + i], b[off + i], p);
}

static auto ntt_core_batch(uint64_t* d_data, int n, uint64_t w, uint64_t p, int batch) -> void {
  int log_n = __builtin_ctz(n);

  static uint64_t* d_tw = nullptr;
  static size_t tw_pool = 0;
  size_t tw_bytes = n * sizeof(uint64_t);
  if (tw_bytes > tw_pool) {
    if (d_tw) cudaFree(d_tw);
    check_cuda(cudaMalloc(&d_tw, tw_bytes));
    tw_pool = tw_bytes;
  }

  constexpr int B = BLOCK_SIZE;
  compute_twiddles<<<(n + B - 1) / B, B>>>(d_tw, w, p, n);
  check_cuda(cudaGetLastError());

  dim3 grid_perm((n + B - 1) / B, batch);
  bit_reverse_permute_batch<<<grid_perm, B>>>(d_data, n, log_n, batch);
  check_cuda(cudaGetLastError());

  dim3 grid_bfly((n / 2 + B - 1) / B, batch);
  for (int stage = 0; stage < log_n; stage++) {
    butterfly_batch<<<grid_bfly, B>>>(d_data, d_tw, stage, n, p, batch);
    check_cuda(cudaGetLastError());
  }

  check_cuda(cudaDeviceSynchronize());
}

auto ntt_forward_batch(uint64_t* d_data, int n, int batch, const NttPrime& prime) -> void {
  uint64_t w = mod_pow_host(prime.g, (prime.p - 1) / n, prime.p);
  ntt_core_batch(d_data, n, w, prime.p, batch);
}

auto ntt_inverse_batch(uint64_t* d_data, int n, int batch, const NttPrime& prime) -> void {
  uint64_t w = mod_pow_host(prime.g, (prime.p - 1) / n, prime.p);
  uint64_t w_inv = mod_pow_host(w, prime.p - 2, prime.p);
  ntt_core_batch(d_data, n, w_inv, prime.p, batch);

  uint64_t n_inv = mod_pow_host((uint64_t)n, prime.p - 2, prime.p);
  constexpr int B = BLOCK_SIZE;
  dim3 grid((n + B - 1) / B, batch);
  scale_mod_batch<<<grid, B>>>(d_data, n, n_inv, prime.p, batch);
  check_cuda(cudaGetLastError());
  check_cuda(cudaDeviceSynchronize());
}

auto ntt_pointwise_mul_batch(uint64_t* d_out, const uint64_t* d_a, const uint64_t* d_b, int n,
                             int batch, uint64_t p) -> void {
  constexpr int B = BLOCK_SIZE;
  dim3 grid((n + B - 1) / B, batch);
  pointwise_mul_batch<<<grid, B>>>(d_out, d_a, d_b, n, p, batch);
  check_cuda(cudaGetLastError());
  check_cuda(cudaDeviceSynchronize());
}
