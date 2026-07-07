#include "bigmul/bigmul.cuh"
#include "bigmul/ntt.cuh"

__global__ auto digit_split(const uint32_t* limbs, uint64_t* digits, int n) -> void {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  digits[2 * i] = limbs[i] & 0xFFFF;
  digits[2 * i + 1] = limbs[i] >> 16;
}

__global__ auto carry_and_assemble(const uint64_t* conv, uint32_t* result, int n, int m) -> void {
  uint64_t carry = 0;
  for (int i = 0; i < 4 * n; i++) {
    uint64_t val = (i < m) ? conv[i] : 0;
    val += carry;
    uint32_t digit = (uint32_t)(val & 0xFFFF);
    carry = val >> 16;
    int limb = i / 2;
    if (i % 2 == 0)
      result[limb] = digit;
    else
      result[limb] |= digit << 16;
  }
}

auto bigmul(const uint32_t* a, const uint32_t* b, uint32_t* result, int n) -> void {
  int n_digits = 2 * n;
  int m = 1;
  while (m < 2 * n_digits) m <<= 1;

  constexpr int B = BLOCK_SIZE;
  const NttPrime& prime = NTT_P1;

  static uint32_t *d_a_raw, *d_b_raw;
  static uint64_t *d_da, *d_db, *d_a, *d_b, *d_conv;
  static uint32_t* d_result;
  static size_t pool_limb = 0, pool_digit = 0;

  size_t limb_bytes = n * sizeof(uint32_t);
  size_t digit_bytes = m * sizeof(uint64_t);

  if (limb_bytes > pool_limb || digit_bytes > pool_digit) {
    if (pool_limb) {
      cudaFree(d_a_raw); cudaFree(d_b_raw); cudaFree(d_da); cudaFree(d_db);
      cudaFree(d_a); cudaFree(d_b); cudaFree(d_conv); cudaFree(d_result);
    }
    check_cuda(cudaMalloc(&d_a_raw, limb_bytes));
    check_cuda(cudaMalloc(&d_b_raw, limb_bytes));
    check_cuda(cudaMalloc(&d_da, digit_bytes));
    check_cuda(cudaMalloc(&d_db, digit_bytes));
    check_cuda(cudaMalloc(&d_a, digit_bytes));
    check_cuda(cudaMalloc(&d_b, digit_bytes));
    check_cuda(cudaMalloc(&d_conv, digit_bytes));
    check_cuda(cudaMalloc(&d_result, 2 * limb_bytes));
    pool_limb = limb_bytes;
    pool_digit = digit_bytes;
  }

  check_cuda(cudaMemcpy(d_a_raw, a, limb_bytes, cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(d_b_raw, b, limb_bytes, cudaMemcpyHostToDevice));

  check_cuda(cudaMemset(d_da, 0, digit_bytes));
  check_cuda(cudaMemset(d_db, 0, digit_bytes));
  digit_split<<<(n + B - 1) / B, B>>>(d_a_raw, d_da, n);
  digit_split<<<(n + B - 1) / B, B>>>(d_b_raw, d_db, n);
  check_cuda(cudaGetLastError());

  check_cuda(cudaMemcpy(d_a, d_da, digit_bytes, cudaMemcpyDeviceToDevice));
  check_cuda(cudaMemcpy(d_b, d_db, digit_bytes, cudaMemcpyDeviceToDevice));

  ntt_forward(d_a, m, prime);
  ntt_forward(d_b, m, prime);
  ntt_inverse_pointwise(d_conv, d_a, d_b, m, prime);

  check_cuda(cudaMemset(d_result, 0, 2 * limb_bytes));
  carry_and_assemble<<<1, 1>>>(d_conv, d_result, n, m);
  check_cuda(cudaGetLastError());
  check_cuda(cudaDeviceSynchronize());

  check_cuda(cudaMemcpy(result, d_result, 2 * limb_bytes, cudaMemcpyDeviceToHost));

}
