#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <random>
#include <vector>

#include "bigmul/bigmul.cuh"

static auto bench(int n, int iters) -> void {
  std::mt19937_64 rng(42 + n);
  std::vector<uint32_t> a(n), b(n), result(2 * n);
  for (int i = 0; i < n; i++) {
    a[i] = (uint32_t)rng();
    b[i] = (uint32_t)rng();
  }

  for (int i = 0; i < iters; i++)
    bigmul(a.data(), b.data(), result.data(), n);
}

auto main(int argc, char** argv) -> int {
  int max_n = 1048576;
  int iters = 3;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--max-n" && i + 1 < argc)
      max_n = std::atoi(argv[++i]);
    else if (arg == "--iters" && i + 1 < argc)
      iters = std::atoi(argv[++i]);
  }

  cudaDeviceProp prop;
  check_cuda(cudaGetDeviceProperties(&prop, 0));
  std::cerr << std::format("device: {} (compute {}.{})\n", prop.name,
                           prop.major, prop.minor);

  for (int n = 64; n <= max_n; n *= 2)
    bench(n, iters);

  return 0;
}
