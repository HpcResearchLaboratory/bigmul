#include <array>
#include <cstdint>
#include <format>
#include <iostream>

#include "bigmul/bigmul.cuh"

auto main() -> int {
  constexpr int n = 4;
  std::array<uint32_t, n> a = {0xFFFFFFFF, 0x00000001, 0x00000000, 0x00000000};
  std::array<uint32_t, n> b = {0x00000002, 0x00000000, 0x00000000, 0x00000000};
  std::array<uint32_t, 2 * n> cpu_result = {};
  std::array<uint32_t, 2 * n> gpu_result = {};

  bigmul_cpu(a.data(), b.data(), cpu_result.data(), n);
  bigmul(a.data(), b.data(), gpu_result.data(), n);

  std::cout << "cpu:";
  for (const auto& r : cpu_result) std::cout << std::format(" {:08X}", r);
  std::cout << '\n';

  std::cout << "gpu:";
  for (const auto& r : gpu_result) std::cout << std::format(" {:08X}", r);
  std::cout << '\n';

  bool match = (cpu_result == gpu_result);
  std::cout << (match ? "MATCH" : "MISMATCH") << '\n';

  return match ? 0 : 1;
}
