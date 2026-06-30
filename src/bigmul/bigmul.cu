#include "bigmul/bigmul.cuh"

#include <cstring>
#include <vector>

__global__ void bigmul_kernel(const uint32_t* a, const uint32_t* b,
                              uint64_t* partial_lo, uint64_t* partial_hi, int n) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= 2 * n) return;

  int j_start = (k >= n) ? (k - n + 1) : 0;
  int j_end = (k < n) ? k : (n - 1);

  uint64_t acc_lo = 0, acc_hi = 0;
  for (int j = j_start; j <= j_end; j++) {
    uint64_t prod = (uint64_t)a[j] * (uint64_t)b[k - j];
    uint64_t prev = acc_lo;
    acc_lo += prod;
    if (acc_lo < prev) acc_hi++;
  }

  partial_lo[k] = acc_lo;
  partial_hi[k] = acc_hi;
}

static void carry_propagate(const uint64_t* partial_lo, const uint64_t* partial_hi,
                            uint32_t* result, int n) {
  uint64_t carry_lo = 0, carry_hi = 0;
  for (int k = 0; k < 2 * n; k++) {
    uint64_t sum_lo = partial_lo[k] + carry_lo;
    uint64_t sum_hi = partial_hi[k] + carry_hi + (sum_lo < partial_lo[k] ? 1 : 0);

    result[k] = (uint32_t)(sum_lo & 0xFFFFFFFF);

    carry_lo = (sum_lo >> 32) | (sum_hi << 32);
    carry_hi = sum_hi >> 32;
  }
}

void bigmul(const uint32_t* a, const uint32_t* b, uint32_t* result, int n) {
  uint32_t *d_a, *d_b;
  uint64_t *d_partial_lo, *d_partial_hi;
  size_t input_bytes = n * sizeof(uint32_t);
  size_t partial_bytes = 2 * n * sizeof(uint64_t);

  CHECK_CUDA(cudaMalloc(&d_a, input_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, input_bytes));
  CHECK_CUDA(cudaMalloc(&d_partial_lo, partial_bytes));
  CHECK_CUDA(cudaMalloc(&d_partial_hi, partial_bytes));

  CHECK_CUDA(cudaMemcpy(d_a, a, input_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b, input_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_partial_lo, 0, partial_bytes));
  CHECK_CUDA(cudaMemset(d_partial_hi, 0, partial_bytes));

  int threads = 256;
  int blocks = (2 * n + threads - 1) / threads;
  bigmul_kernel<<<blocks, threads>>>(d_a, d_b, d_partial_lo, d_partial_hi, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<uint64_t> h_partial_lo(2 * n);
  std::vector<uint64_t> h_partial_hi(2 * n);
  CHECK_CUDA(cudaMemcpy(h_partial_lo.data(), d_partial_lo, partial_bytes, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_partial_hi.data(), d_partial_hi, partial_bytes, cudaMemcpyDeviceToHost));

  carry_propagate(h_partial_lo.data(), h_partial_hi.data(), result, n);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_partial_lo));
  CHECK_CUDA(cudaFree(d_partial_hi));
}

void bigmul_cpu(const uint32_t* a, const uint32_t* b, uint32_t* result, int n) {
  memset(result, 0, 2 * n * sizeof(uint32_t));

  for (int i = 0; i < n; i++) {
    uint64_t carry = 0;
    for (int j = 0; j < n; j++) {
      __uint128_t prod = (__uint128_t)a[i] * b[j] + result[i + j] + carry;
      result[i + j] = (uint32_t)prod;
      carry = (uint64_t)(prod >> 32);
    }
    int k = i + n;
    while (carry && k < 2 * n) {
      uint64_t sum = (uint64_t)result[k] + carry;
      result[k] = (uint32_t)sum;
      carry = sum >> 32;
      k++;
    }
  }
}
