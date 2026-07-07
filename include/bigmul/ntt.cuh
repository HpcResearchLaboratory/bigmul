#pragma once

#include <cstdint>

struct NttPrime {
  uint64_t p;
  uint64_t g;
};

// Goldilocks prime: p = 2^64 - 2^32 + 1, p - 1 = 2^32 * (2^32 - 1).
// Supports NTT sizes up to 2^32 and avoids the need for CRT across
// multiple smaller primes.
constexpr NttPrime NTT_P1 = {0xFFFFFFFF00000001ULL, 7ULL};

auto mod_pow_host(uint64_t base, uint64_t exp, uint64_t p) -> uint64_t;

auto ntt_forward(uint64_t* d_data, int n, const NttPrime& prime) -> void;
auto ntt_inverse(uint64_t* d_data, int n, const NttPrime& prime) -> void;
auto ntt_pointwise_mul(uint64_t* d_out, const uint64_t* d_a, const uint64_t* d_b, int n,
                       uint64_t p) -> void;

// Batched variants: d_data/d_out/d_a/d_b hold `batch` contiguous blocks of
// n elements each. The twiddle table (depends only on n) is computed once
// and shared across the whole batch.
auto ntt_forward_batch(uint64_t* d_data, int n, int batch, const NttPrime& prime) -> void;
auto ntt_inverse_batch(uint64_t* d_data, int n, int batch, const NttPrime& prime) -> void;
auto ntt_pointwise_mul_batch(uint64_t* d_out, const uint64_t* d_a, const uint64_t* d_b, int n,
                             int batch, uint64_t p) -> void;
