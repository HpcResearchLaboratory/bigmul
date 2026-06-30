#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <string>
#include <vector>

#include "bigmul/bigmul.cuh"

static auto hex_to_limbs(const std::string& hex) -> std::vector<uint32_t> {
  std::vector<uint32_t> limbs;
  int len = hex.size();
  for (int i = len; i > 0; i -= 8) {
    int start = (i >= 8) ? i - 8 : 0;
    int count = i - start;
    limbs.push_back((uint32_t)strtoul(hex.substr(start, count).c_str(), nullptr, 16));
  }
  return limbs;
}

static auto limbs_to_hex(const uint32_t* limbs, int n) -> std::string {
  int top = n - 1;
  while (top > 0 && limbs[top] == 0) top--;

  std::string out = std::format("{:X}", limbs[top]);
  for (int i = top - 1; i >= 0; i--)
    out += std::format("{:08X}", limbs[i]);
  return out;
}

auto main(int argc, char** argv) -> int {
  if (argc != 3) {
    std::cerr << "usage: multiply <hex_a> <hex_b>\n";
    return 1;
  }

  auto a = hex_to_limbs(argv[1]);
  auto b = hex_to_limbs(argv[2]);

  int n = (int)std::max(a.size(), b.size());
  a.resize(n, 0);
  b.resize(n, 0);

  std::vector<uint32_t> result(2 * n, 0);
  bigmul(a.data(), b.data(), result.data(), n);

  std::cout << limbs_to_hex(result.data(), 2 * n) << '\n';
  return 0;
}
