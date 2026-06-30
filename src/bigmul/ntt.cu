#include "bigmul/ntt.cuh"
#include "bigmul/bigmul.cuh"

#include <vector>

__device__ uint32_t mod_mul(uint32_t a, uint32_t b, uint32_t p) {
  return (uint64_t)a * b % p;
}

__device__ uint32_t mod_add(uint32_t a, uint32_t b, uint32_t p) {
  uint32_t r = a + b;
  return r >= p ? r - p : r;
}

__device__ uint32_t mod_sub(uint32_t a, uint32_t b, uint32_t p) {
  return a >= b ? a - b : a + p - b;
}

uint32_t mod_pow_host(uint32_t base, uint32_t exp, uint32_t p) {
  uint64_t result = 1, b = base;
  while (exp > 0) {
    if (exp & 1) result = result * b % p;
    b = b * b % p;
    exp >>= 1;
  }
  return (uint32_t)result;
}

__global__ void bit_reverse_permute(uint32_t* data, int n, int log_n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  int rev = 0, x = i;
  for (int j = 0; j < log_n; j++) {
    rev = (rev << 1) | (x & 1);
    x >>= 1;
  }

  if (i < rev) {
    uint32_t tmp = data[i];
    data[i] = data[rev];
    data[rev] = tmp;
  }
}

__global__ void butterfly(uint32_t* data, const uint32_t* twiddles, int stage,
                          int n, uint32_t p) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n / 2) return;

  int half = 1 << stage;
  int group = k / half;
  int pos = k % half;
  int i = group * 2 * half + pos;
  int j = i + half;
  int step = n >> (stage + 1);

  uint32_t tw = twiddles[step * pos];
  uint32_t u = data[i];
  uint32_t v = mod_mul(data[j], tw, p);
  data[i] = mod_add(u, v, p);
  data[j] = mod_sub(u, v, p);
}

__global__ void scale_mod(uint32_t* data, int n, uint32_t factor, uint32_t p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  data[i] = mod_mul(data[i], factor, p);
}

__global__ void pointwise_mul(uint32_t* out, const uint32_t* a, const uint32_t* b,
                              int n, uint32_t p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = mod_mul(a[i], b[i], p);
}

static void ntt_core(uint32_t* d_data, int n, uint32_t w, uint32_t p) {
  int log_n = __builtin_ctz(n);

  std::vector<uint32_t> tw(n);
  tw[0] = 1;
  for (int i = 1; i < n; i++)
    tw[i] = (uint64_t)tw[i - 1] * w % p;

  uint32_t* d_tw;
  CHECK_CUDA(cudaMalloc(&d_tw, n * sizeof(uint32_t)));
  CHECK_CUDA(
      cudaMemcpy(d_tw, tw.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));

  int threads = 256;
  bit_reverse_permute<<<(n + threads - 1) / threads, threads>>>(d_data, n,
                                                                 log_n);
  CHECK_CUDA(cudaGetLastError());

  for (int stage = 0; stage < log_n; stage++) {
    butterfly<<<(n / 2 + threads - 1) / threads, threads>>>(d_data, d_tw, stage,
                                                             n, p);
    CHECK_CUDA(cudaGetLastError());
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaFree(d_tw));
}

void ntt_forward(uint32_t* d_data, int n, const NttPrime& prime) {
  uint32_t w = mod_pow_host(prime.g, (prime.p - 1) / n, prime.p);
  ntt_core(d_data, n, w, prime.p);
}

void ntt_inverse(uint32_t* d_data, int n, const NttPrime& prime) {
  uint32_t w = mod_pow_host(prime.g, (prime.p - 1) / n, prime.p);
  uint32_t w_inv = mod_pow_host(w, prime.p - 2, prime.p);
  ntt_core(d_data, n, w_inv, prime.p);

  uint32_t n_inv = mod_pow_host((uint32_t)n, prime.p - 2, prime.p);
  int threads = 256;
  scale_mod<<<(n + threads - 1) / threads, threads>>>(d_data, n, n_inv,
                                                       prime.p);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

void ntt_pointwise_mul(uint32_t* d_out, const uint32_t* d_a,
                       const uint32_t* d_b, int n, uint32_t p) {
  int threads = 256;
  pointwise_mul<<<(n + threads - 1) / threads, threads>>>(d_out, d_a, d_b, n,
                                                           p);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}
