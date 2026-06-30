#include "bigmul/bigmul.cuh"
#include "bigmul/ntt.cuh"

#include <vector>

void bigmul(const uint32_t* a, const uint32_t* b, uint32_t* result, int n) {
  int n_digits = 2 * n;
  int m = 1;
  while (m < 2 * n_digits) m <<= 1;

  std::vector<uint32_t> da(m, 0), db(m, 0);
  for (int i = 0; i < n; i++) {
    da[2 * i] = a[i] & 0xFFFF;
    da[2 * i + 1] = a[i] >> 16;
    db[2 * i] = b[i] & 0xFFFF;
    db[2 * i + 1] = b[i] >> 16;
  }

  const NttPrime primes[] = {NTT_P1, NTT_P2, NTT_P3};
  std::vector<uint32_t> res[3];

  uint32_t *d_a, *d_b, *d_c;
  size_t bytes = m * sizeof(uint32_t);
  CHECK_CUDA(cudaMalloc(&d_a, bytes));
  CHECK_CUDA(cudaMalloc(&d_b, bytes));
  CHECK_CUDA(cudaMalloc(&d_c, bytes));

  for (int pi = 0; pi < 3; pi++) {
    CHECK_CUDA(cudaMemcpy(d_a, da.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, db.data(), bytes, cudaMemcpyHostToDevice));

    ntt_forward(d_a, m, primes[pi]);
    ntt_forward(d_b, m, primes[pi]);
    ntt_pointwise_mul(d_c, d_a, d_b, m, primes[pi].p);
    ntt_inverse(d_c, m, primes[pi]);

    res[pi].resize(m);
    CHECK_CUDA(
        cudaMemcpy(res[pi].data(), d_c, bytes, cudaMemcpyDeviceToHost));
  }

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));

  uint32_t p1 = NTT_P1.p, p2 = NTT_P2.p, p3 = NTT_P3.p;
  uint32_t p1_inv_p2 = mod_pow_host(p1, p2 - 2, p2);
  uint32_t p1_inv_p3 = mod_pow_host(p1, p3 - 2, p3);
  uint32_t p2_inv_p3 = mod_pow_host(p2, p3 - 2, p3);

  int out_digits = 4 * n;
  uint64_t carry = 0;

  for (int i = 0; i < out_digits; i++) {
    uint64_t r1 = (i < m) ? res[0][i] : 0;
    uint64_t r2 = (i < m) ? res[1][i] : 0;
    uint64_t r3 = (i < m) ? res[2][i] : 0;

    uint64_t a1 = r1;
    uint64_t diff2 = (r2 + p2 - a1 % p2) % p2;
    uint64_t a2 = diff2 * p1_inv_p2 % p2;
    uint64_t diff3 = (r3 + p3 - a1 % p3) % p3;
    uint64_t tmp = diff3 * p1_inv_p3 % p3;
    uint64_t a3 = (tmp + p3 - a2 % p3) % p3 * p2_inv_p3 % p3;

    __uint128_t x = a1 + (__uint128_t)a2 * p1 + (__uint128_t)a3 * p1 * p2;
    x += carry;

    uint32_t digit = (uint32_t)(x & 0xFFFF);
    carry = (uint64_t)(x >> 16);

    int limb = i / 2;
    if (limb < 2 * n) {
      if (i % 2 == 0)
        result[limb] = digit;
      else
        result[limb] |= digit << 16;
    }
  }
}
