#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static constexpr int MAX_GPU_BATCH = 128;

// Runs up to MAX_GPU_BATCH pending pairs through a single bigmul_batch()
// call: all pairs are padded to the batch's largest limb count so every
// kernel launch handles the whole group in one shot instead of one launch
// per pair.
static auto flush_batch(std::vector<std::string>& pending_a, std::vector<std::string>& pending_b)
    -> void {
  int batch = (int)pending_a.size();
  if (batch == 0) return;

  std::vector<std::vector<uint32_t>> limbs_a(batch), limbs_b(batch);
  size_t n = 1;
  for (int i = 0; i < batch; i++) {
    limbs_a[i] = hex_to_limbs(pending_a[i]);
    limbs_b[i] = hex_to_limbs(pending_b[i]);
    n = std::max({n, limbs_a[i].size(), limbs_b[i].size()});
  }

  std::vector<uint32_t> a_buf((size_t)batch * n, 0), b_buf((size_t)batch * n, 0);
  std::vector<uint32_t> result_buf((size_t)batch * 2 * n, 0);
  for (int i = 0; i < batch; i++) {
    std::copy(limbs_a[i].begin(), limbs_a[i].end(), a_buf.begin() + (size_t)i * n);
    std::copy(limbs_b[i].begin(), limbs_b[i].end(), b_buf.begin() + (size_t)i * n);
  }

  bigmul_batch(a_buf.data(), b_buf.data(), result_buf.data(), (int)n, batch);

  for (int i = 0; i < batch; i++)
    std::cout << limbs_to_hex(result_buf.data() + (size_t)i * 2 * n, 2 * n) << '\n';

  pending_a.clear();
  pending_b.clear();
}

static auto run_batch() -> int {
  std::string a, b;
  std::vector<std::string> pending_a, pending_b;
  while (std::getline(std::cin, a) && std::getline(std::cin, b)) {
    pending_a.push_back(std::move(a));
    pending_b.push_back(std::move(b));
    if ((int)pending_a.size() == MAX_GPU_BATCH) flush_batch(pending_a, pending_b);
  }
  flush_batch(pending_a, pending_b);
  return 0;
}

static auto run_binary() -> int {
  uint32_t n;
  while (fread(&n, sizeof(n), 1, stdin) == 1) {
    std::vector<uint32_t> a(n), b(n), result(2 * n, 0);
    fread(a.data(), sizeof(uint32_t), n, stdin);
    fread(b.data(), sizeof(uint32_t), n, stdin);
    bigmul(a.data(), b.data(), result.data(), n);
    fwrite(result.data(), sizeof(uint32_t), 2 * n, stdout);
  }
  return 0;
}

static auto run_chain(std::vector<std::string>& args) -> int {
  auto acc = hex_to_limbs(args[0]);
  for (size_t i = 1; i < args.size(); i++) {
    auto b = hex_to_limbs(args[i]);
    auto n = std::max(acc.size(), b.size());
    acc.resize(n, 0);
    b.resize(n, 0);
    std::vector<uint32_t> result(2 * n, 0);
    bigmul(acc.data(), b.data(), result.data(), n);
    while (result.size() > 1 && result.back() == 0) result.pop_back();
    acc = std::move(result);
  }
  std::cout << limbs_to_hex(acc.data(), (int)acc.size()) << '\n';
  return 0;
}

auto main(int argc, char** argv) -> int {
  if (argc >= 2 && strcmp(argv[1], "--batch") == 0)
    return run_batch();
  if (argc >= 2 && strcmp(argv[1], "--binary") == 0)
    return run_binary();

  std::vector<std::string> args;
  if (argc >= 3) {
    for (int i = 1; i < argc; i++) args.emplace_back(argv[i]);
  } else {
    std::string line;
    while (std::getline(std::cin, line))
      if (!line.empty()) args.push_back(std::move(line));
  }

  if (args.size() < 2) {
    std::cerr << "usage: multiply [--batch] <hex> <hex> [<hex>...]\n"
              << "       multiply --batch < pairs.txt\n";
    return 1;
  }

  return run_chain(args);
}
