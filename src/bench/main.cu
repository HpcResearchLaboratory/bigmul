#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <random>
#include <vector>

#include "bigmul/bigmul.cuh"

static void bench(int n, int warmup, int iters) {
  std::mt19937_64 rng(42 + n);
  std::vector<uint32_t> a(n), b(n), result(2 * n);
  for (int i = 0; i < n; i++) {
    a[i] = (uint32_t)rng();
    b[i] = (uint32_t)rng();
  }

  for (int i = 0; i < warmup; i++)
    bigmul(a.data(), b.data(), result.data(), n);

  cudaEvent_t start, stop;
  check_cuda(cudaEventCreate(&start));
  check_cuda(cudaEventCreate(&stop));

  check_cuda(cudaEventRecord(start));
  for (int i = 0; i < iters; i++)
    bigmul(a.data(), b.data(), result.data(), n);
  check_cuda(cudaEventRecord(stop));
  check_cuda(cudaEventSynchronize(stop));

  float total_ms = 0;
  check_cuda(cudaEventElapsedTime(&total_ms, start, stop));
  float avg_ms = total_ms / iters;

  std::cout << std::format("{},{:.4f}\n", n, avg_ms);

  check_cuda(cudaEventDestroy(start));
  check_cuda(cudaEventDestroy(stop));
}

auto main(int argc, char** argv) -> int {
  int max_n = 1048576;
  int warmup = 2;
  int iters = 5;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--max-n" && i + 1 < argc)
      max_n = std::atoi(argv[++i]);
    else if (arg == "--warmup" && i + 1 < argc)
      warmup = std::atoi(argv[++i]);
    else if (arg == "--iters" && i + 1 < argc)
      iters = std::atoi(argv[++i]);
  }

  cudaDeviceProp prop;
  check_cuda(cudaGetDeviceProperties(&prop, 0));
  std::cerr << std::format("device: {} (compute {}.{})\n", prop.name,
                           prop.major, prop.minor);

  std::cout << "n,avg_ms\n";
  for (int n = 64; n <= max_n; n *= 2)
    bench(n, warmup, iters);

  return 0;
}
