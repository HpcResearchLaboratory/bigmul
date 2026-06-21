#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <random>
#include <vector>

#include "bigmul/bigmul.cuh"

static int tests_run = 0;
static int tests_passed = 0;

static bool check(const char* name, const uint32_t* a, const uint32_t* b, int n) {
  std::vector<uint32_t> cpu(2 * n, 0);
  std::vector<uint32_t> gpu(2 * n, 0);

  bigmul_cpu(a, b, cpu.data(), n);
  bigmul(a, b, gpu.data(), n);

  tests_run++;
  if (memcmp(cpu.data(), gpu.data(), 2 * n * sizeof(uint32_t)) == 0) {
    tests_passed++;
    std::cout << std::format("  PASS  {}\n", name);
    return true;
  }

  std::cout << std::format("  FAIL  {}\n", name);
  std::cout << "    cpu:";
  for (int i = 0; i < 2 * n; i++) std::cout << std::format(" {:08X}", cpu[i]);
  std::cout << "\n    gpu:";
  for (int i = 0; i < 2 * n; i++) std::cout << std::format(" {:08X}", gpu[i]);
  std::cout << '\n';
  return false;
}

static void test_single_limb() {
  uint32_t a[] = {7};
  uint32_t b[] = {6};
  check("single limb 7*6", a, b, 1);

  uint32_t c[] = {0xFFFFFFFF};
  uint32_t d[] = {0xFFFFFFFF};
  check("single limb max*max", c, d, 1);

  uint32_t e[] = {0};
  uint32_t f[] = {12345};
  check("single limb 0*x", e, f, 1);
}

static void test_powers_of_two() {
  uint32_t a[4] = {0, 1, 0, 0};
  uint32_t b[4] = {0, 0, 1, 0};
  check("powers of two 2^32 * 2^64", a, b, 4);
}

static void test_all_ones() {
  constexpr int n = 8;
  uint32_t a[n], b[n];
  for (int i = 0; i < n; i++) a[i] = b[i] = 0xFFFFFFFF;
  check("all ones 8 limbs", a, b, n);
}

static void test_zero_multiplicand() {
  constexpr int n = 4;
  uint32_t a[n] = {0x12345678, 0xDEADBEEF, 0xCAFEBABE, 0x01020304};
  uint32_t b[n] = {};
  check("zero multiplicand", a, b, n);
}

static void test_identity() {
  constexpr int n = 4;
  uint32_t a[n] = {0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD};
  uint32_t b[n] = {1, 0, 0, 0};
  check("multiply by one", a, b, n);
}

static void test_random(int n, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<uint32_t> a(n), b(n);
  for (int i = 0; i < n; i++) {
    a[i] = (uint32_t)rng();
    b[i] = (uint32_t)rng();
  }
  check(std::format("random n={} seed={}", n, seed).c_str(), a.data(), b.data(), n);
}

auto main() -> int {
  std::cout << "bigmul correctness tests\n";

  test_single_limb();
  test_powers_of_two();
  test_all_ones();
  test_zero_multiplicand();
  test_identity();

  for (int n : {1, 4, 16, 64, 256, 1024}) {
    test_random(n, 42 + n);
  }

  std::cout << std::format("\n{}/{} passed\n", tests_passed, tests_run);
  return (tests_passed == tests_run) ? 0 : 1;
}
