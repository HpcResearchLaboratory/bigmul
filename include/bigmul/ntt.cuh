#pragma once

#include <cstdint>

struct NttPrime {
  uint32_t p;
  uint32_t g;
};

constexpr NttPrime NTT_P1 = {998244353, 3};
constexpr NttPrime NTT_P2 = {469762049, 3};
constexpr NttPrime NTT_P3 = {754974721, 11};

auto mod_pow_host(uint32_t base, uint32_t exp, uint32_t p) -> uint32_t;

auto ntt_forward(uint32_t* d_data, int n, const NttPrime& prime) -> void;
auto ntt_inverse(uint32_t* d_data, int n, const NttPrime& prime) -> void;
auto ntt_pointwise_mul(uint32_t* d_out, const uint32_t* d_a, const uint32_t* d_b,
                       int n, uint32_t p) -> void;
