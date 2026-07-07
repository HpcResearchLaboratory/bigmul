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

// Multiplies `batch` independent pairs in one shot, sharing every kernel
// launch (and NTT twiddle table) across the whole batch instead of issuing
// them once per pair. All pairs must share the same limb count n (pad with
// zero limbs on the host if needed). a/b are batch*n limbs laid out as
// batch contiguous blocks of n limbs each; result is batch*2*n limbs laid
// out the same way. batch must be <= 128.
auto bigmul_batch(const uint32_t* a, const uint32_t* b, uint32_t* result, int n, int batch) -> void;
