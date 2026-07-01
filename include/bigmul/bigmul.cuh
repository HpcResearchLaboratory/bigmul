#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <source_location>

[[gnu::always_inline]] inline auto check_cuda(
    cudaError_t err, std::source_location loc = std::source_location::current()) -> void {
  if (err == cudaSuccess) return;

  fprintf(stderr, "CUDA error at %s:%u: %s\n", loc.file_name(), loc.line(),
          cudaGetErrorString(err));
  std::exit(1);
}

constexpr int BLOCK_SIZE = 256;

auto bigmul(const uint32_t* a, const uint32_t* b, uint32_t* result, int n) -> void;
