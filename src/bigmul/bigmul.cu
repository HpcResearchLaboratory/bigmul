#include "bigmul/bigmul.cuh"

#include <cstring>

__global__ void bigmul_kernel(const uint32_t* a, const uint32_t* b, uint32_t* result, int n) {
  // TODO: implement multiplication with carry propagation
}

void bigmul(const uint32_t* a, const uint32_t* b, uint32_t* result, int n) {
  uint32_t *d_a, *d_b, *d_result;
  size_t bytes = n * sizeof(uint32_t);

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_result, 2 * n * sizeof(uint32_t));

  cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);
  cudaMemset(d_result, 0, 2 * n * sizeof(uint32_t));

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  bigmul_kernel<<<blocks, threads>>>(d_a, d_b, d_result, n);

  cudaMemcpy(result, d_result, 2 * n * sizeof(uint32_t), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_result);
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
