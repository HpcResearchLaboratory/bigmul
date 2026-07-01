#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

  char buf[16];
  snprintf(buf, sizeof(buf), "%X", limbs[top]);
  std::string out = buf;
  for (int i = top - 1; i >= 0; i--) {
    snprintf(buf, sizeof(buf), "%08X", limbs[i]);
    out += buf;
  }
  return out;
}

static auto mul(std::vector<uint32_t>& a, const std::vector<uint32_t>& b) -> void {
  auto n = std::max(a.size(), b.size());
  a.resize(n, 0);
  auto b_padded = b;
  b_padded.resize(n, 0);

  std::vector<uint32_t> result(2 * n, 0);
  bigmul(a.data(), b_padded.data(), result.data(), n);

  while (result.size() > 1 && result.back() == 0) result.pop_back();
  a = std::move(result);
}

auto main(int argc, char** argv) -> int {
  std::vector<std::string> args;

  if (argc >= 3) {
    for (int i = 1; i < argc; i++) args.emplace_back(argv[i]);
  } else {
    std::string line;
    while (std::getline(std::cin, line))
      if (!line.empty()) args.push_back(std::move(line));
  }

  if (args.size() < 2) {
    std::cerr << "usage: multiply <hex> <hex> [<hex>...]\n"
              << "       cat a.hex b.hex | multiply\n";
    return 1;
  }

  auto acc = hex_to_limbs(args[0]);
  for (size_t i = 1; i < args.size(); i++) mul(acc, hex_to_limbs(args[i]));

  std::cout << limbs_to_hex(acc.data(), (int)acc.size()) << '\n';
  return 0;
}
