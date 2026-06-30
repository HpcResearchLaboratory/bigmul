#pragma once

#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <source_location>

[[gnu::always_inline]] inline auto check_cuda(
    cudaError_t err, std::source_location loc = std::source_location::current()) -> void {
  if (err == cudaSuccess) return;

  std::cerr << std::format("CUDA error at {}:{}: {}\n", loc.file_name(), loc.line(),
                           cudaGetErrorString(err));
  std::exit(1);
}

auto bigmul(const uint32_t* a, const uint32_t* b, uint32_t* result, int n) -> void;
