#include "bigmul/bigmul.cuh"
#include "bigmul/ntt.cuh"

#include <vector>

__global__ auto crt_kernel(const uint32_t* r1, const uint32_t* r2, const uint32_t* r3,
                           uint64_t* conv, int m, uint32_t p1, uint32_t p2, uint32_t p3,
                           uint32_t p1_inv_p2, uint32_t p1_inv_p3, uint32_t p2_inv_p3) -> void {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;

  uint64_t a1 = r1[i];
  uint64_t diff2 = ((uint64_t)r2[i] + p2 - a1 % p2) % p2;
  uint64_t a2 = diff2 * p1_inv_p2 % p2;
  uint64_t diff3 = ((uint64_t)r3[i] + p3 - a1 % p3) % p3;
  uint64_t tmp = diff3 * p1_inv_p3 % p3;
  uint64_t a3 = (tmp + p3 - a2 % p3) % p3 * p2_inv_p3 % p3;

  conv[i] = a1 + a2 * (uint64_t)p1 + a3 * (uint64_t)p1 * (uint64_t)p2;
}

static auto carry_propagate(const uint64_t* conv, uint32_t* result, int n, int m) -> void {
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

  std::vector<uint32_t> da(m, 0), db(m, 0);
  for (int i = 0; i < n; i++) {
    da[2 * i] = a[i] & 0xFFFF;
    da[2 * i + 1] = a[i] >> 16;
    db[2 * i] = b[i] & 0xFFFF;
    db[2 * i + 1] = b[i] >> 16;
  }

  constexpr int P = 3;
  const NttPrime primes[P] = {NTT_P1, NTT_P2, NTT_P3};

  uint32_t *d_a, *d_b, *d_c[P];
  uint64_t* d_conv;
  size_t bytes = m * sizeof(uint32_t);

  check_cuda(cudaMalloc(&d_a, bytes));
  check_cuda(cudaMalloc(&d_b, bytes));
  for (int i = 0; i < P; i++)
    check_cuda(cudaMalloc(&d_c[i], bytes));
  check_cuda(cudaMalloc(&d_conv, m * sizeof(uint64_t)));

  for (int pi = 0; pi < P; pi++) {
    check_cuda(cudaMemcpy(d_a, da.data(), bytes, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_b, db.data(), bytes, cudaMemcpyHostToDevice));

    ntt_forward(d_a, m, primes[pi]);
    ntt_forward(d_b, m, primes[pi]);
    ntt_pointwise_mul(d_c[pi], d_a, d_b, m, primes[pi].p);
    ntt_inverse(d_c[pi], m, primes[pi]);
  }

  uint32_t p1 = NTT_P1.p, p2 = NTT_P2.p, p3 = NTT_P3.p;
  uint32_t p1_inv_p2 = mod_pow_host(p1, p2 - 2, p2);
  uint32_t p1_inv_p3 = mod_pow_host(p1, p3 - 2, p3);
  uint32_t p2_inv_p3 = mod_pow_host(p2, p3 - 2, p3);

  constexpr int B = BLOCK_SIZE;
  crt_kernel<<<(m + B - 1) / B, B>>>(d_c[0], d_c[1], d_c[2], d_conv, m, p1, p2, p3, p1_inv_p2,
                                      p1_inv_p3, p2_inv_p3);
  check_cuda(cudaGetLastError());
  check_cuda(cudaDeviceSynchronize());

  std::vector<uint64_t> h_conv(m);
  check_cuda(cudaMemcpy(h_conv.data(), d_conv, m * sizeof(uint64_t), cudaMemcpyDeviceToHost));

  carry_propagate(h_conv.data(), result, n, m);

  check_cuda(cudaFree(d_a));
  check_cuda(cudaFree(d_b));
  for (int i = 0; i < P; i++)
    check_cuda(cudaFree(d_c[i]));
  check_cuda(cudaFree(d_conv));
}
