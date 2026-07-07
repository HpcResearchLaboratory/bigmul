# Big Number Multiplication on GPU — Baseline

## Problem

Multiply two arbitrarily large integers on a GPU using CUDA. The numbers
are represented as arrays of `uint32_t` limbs in little-endian order
(least significant limb at index 0). Two n-limb inputs produce a 2n-limb
result.

## Why limbs

CPUs and GPUs work with fixed-width integers — 32 or 64 bits. A number
with millions of digits doesn't fit in any hardware register. So we
split it into an array of fixed-width "limbs", each holding a chunk of
the number.

We use `uint32_t` limbs in little-endian order: limb 0 holds the least
significant 32 bits, limb 1 the next 32 bits, and so on. The number
`0x1FFFFFFFF` (8,589,934,591) is stored as `{0xFFFFFFFF, 0x00000001}`.

Why uint32 instead of uint64? The NTT butterfly does modular
multiplication: `(uint64_t)a * b % p`. With 32-bit values and a 30-bit
prime, the intermediate product fits in 64 bits. Using uint64 limbs
would require 128-bit intermediates, which CUDA doesn't support
natively.

Why little-endian? Carry propagation flows from low to high — the
natural direction for array index 0 → n. It also matches how
hardware addition works.

## Why NTT

Schoolbook multiplication is O(n²) — each output position sums all
cross-products of input limbs. It parallelizes well (each output column
is independent) but the quadratic work makes it impractical for large
inputs.

NTT-based multiplication reduces this to O(n log n) by converting the
problem into a convolution. The key insight: multiplying two numbers is
equivalent to convolving their digit representations, and convolution
in the frequency domain is just pointwise multiplication.

The Number Theoretic Transform (NTT) is the integer analog of the Fast
Fourier Transform (FFT). It works in modular arithmetic (mod a prime p)
instead of complex numbers, so there are no floating-point rounding
errors — the result is exact.

## Why 3 primes

A single NTT prime can't handle large inputs. The convolution at each
output position sums up to n products of digit pairs. For base-2^16
digits with n up to 2M, the maximum convolution value is:

    n_digits × (2^16 - 1)^2 ≈ 2^21 × 2^32 = 2^53

A single 30-bit prime (like 998244353) can only represent values up to
~10^9 — the convolution overflows.

Solution: compute the convolution modulo 3 different primes, then use
the Chinese Remainder Theorem (CRT) to reconstruct the true value. The
product of the 3 primes (~2^88) is larger than the maximum convolution
value (2^53), so CRT gives the exact answer.

The primes:
- p1 = 998244353 = 119 × 2^23 + 1 (primitive root: 3)
- p2 = 469762049 = 7 × 2^26 + 1 (primitive root: 3)
- p3 = 754974721 = 45 × 2^24 + 1 (primitive root: 11)

These are "NTT-friendly" — each p-1 has a large power-of-2 factor,
which means primitive roots of unity of high order exist, enabling
NTT of size up to 2^23 (8M elements).

## Data flow

```
Input: a[0..n-1], b[0..n-1] (uint32 limbs, little-endian)

1. Digit splitting (CPU)
   Each uint32 limb → two base-2^16 digits (lo and hi 16 bits)
   n limbs → 2n digits
   Pad to NTT length m = next power of 2 ≥ 4n

2. For each prime p (3 passes):
   a. Upload digit arrays to GPU
   b. Forward NTT on both arrays
   c. Pointwise multiply (element-wise, mod p)
   d. Inverse NTT on the product
   e. Download result to host

3. CRT — Garner's algorithm (CPU)
   Combine the 3 mod-p results into the true convolution value
   at each position using the Chinese Remainder Theorem

4. Carry propagation (CPU)
   Walk the convolution values left-to-right, extract base-2^16
   digit, propagate carry to next position

5. Limb assembly (CPU)
   Pair up base-2^16 digits back into uint32 limbs

Output: result[0..2n-1] (uint32 limbs)
```

## NTT implementation

### Iterative Cooley-Tukey (decimation-in-time)

The NTT decomposes into log2(m) "butterfly" stages. Each stage pairs
elements at a specific stride and combines them using a twiddle factor
(a power of the primitive root of unity).

```
Stage 0: stride 1   — pairs (0,1), (2,3), (4,5), ...
Stage 1: stride 2   — pairs (0,2), (1,3), (4,6), ...
Stage k: stride 2^k — pairs separated by 2^k elements
```

Before the butterfly stages, the input is rearranged in bit-reversed
order. This ensures the final output is in natural order.

### GPU kernels

**bit_reverse_permute** — one thread per element, swaps element i with
element bit_reverse(i). Guard `if (i < rev)` ensures each swap happens
once.

**butterfly** — one thread per butterfly operation (n/2 threads per
stage). Each thread reads two elements, multiplies one by the twiddle
factor, adds/subtracts, writes back. One kernel launch per stage.

**pointwise_mul** — element-wise modular multiplication, one thread
per element.

**scale_mod** — multiply all elements by n^-1 mod p (part of inverse
NTT normalization).

### Modular arithmetic

All operations are mod p (a 30-bit prime). Modular multiply uses a
64-bit intermediate: `(uint64_t)a * b % p`. Add and subtract use
conditional correction to stay in [0, p).

### Twiddle factors

Precomputed on the CPU as powers of the primitive root:
`tw[i] = g^((p-1)/m × i) mod p`. Uploaded to GPU before each NTT.
The butterfly kernel looks up `twiddles[step * pos]` per thread.

## Worked example: 0x1FFFFFFFF × 0x2

Input: a = 8,589,934,591, b = 2. Expected result: 17,179,869,182 = 0x3FFFFFFFE.

As uint32 limbs (little-endian):
```
a = {0xFFFFFFFF, 0x00000001}    (n = 2 limbs)
b = {0x00000002, 0x00000000}
```

### Step 1: Digit splitting

Each uint32 limb → two base-2^16 digits (lo and hi 16 bits):
```
a digits = {0xFFFF, 0xFFFF, 0x0001, 0x0000}
b digits = {0x0002, 0x0000, 0x0000, 0x0000}
```

NTT length: n_digits = 4, need m ≥ 2 × 4 = 8, so m = 8.
Pad both to length 8 with zeros.

### Step 2: Forward NTT (for each prime)

Taking p1 = 998244353 as example. The primitive 8th root of unity is
w = 3^(998244352/8) mod 998244353 = 3^124780544 mod 998244353.

The forward NTT transforms the digit array from the "time domain"
(digit values) to the "frequency domain" (evaluations of the digit
polynomial at powers of w). This is where bit-reversal and the 3
butterfly stages (log2(8) = 3) happen.

After forward NTT, both arrays are in the frequency domain.

### Step 3: Pointwise multiply

In the frequency domain, convolution becomes element-wise
multiplication:
```
C_freq[i] = A_freq[i] × B_freq[i] mod p    (for each i = 0..7)
```

This is why NTT is powerful — it turns O(n²) convolution into
O(n) pointwise operations (after the O(n log n) transform).

### Step 4: Inverse NTT

Transform the product back from frequency domain to time domain.
Same butterfly structure but with the inverse root w^-1, and
multiply everything by 8^-1 mod p at the end.

The result is the convolution of the original digit arrays mod p:
```
conv[k] = Σ a_digits[i] × b_digits[k-i]    (mod p)
```

For our example, the true (un-modded) convolution is:
```
conv[0] = 0xFFFF × 0x0002           = 0x1FFFE  =  131070
conv[1] = 0xFFFF × 0x0000 +
          0xFFFF × 0x0002           = 0x1FFFE  =  131070
conv[2] = 0xFFFF × 0x0000 +
          0xFFFF × 0x0000 +
          0x0001 × 0x0002           = 0x00002  =       2
conv[3] = 0x0001 × 0x0000 + ...     = 0x00000  =       0
conv[4..7]                           = 0
```

These values are all small (< p), so mod p gives the same values.
For large inputs, the convolution values can be up to ~2^53 — larger
than any single prime. That's why we need 3 primes.

### Step 5: CRT (Garner's algorithm)

Since all three primes give the same result in this small example,
CRT trivially returns the same values. For large inputs, CRT
combines three different mod-p residues into the true value using:

```
x = a1 + a2 × p1 + a3 × p1 × p2
```

where a1, a2, a3 are computed from the residues via modular inverses.

### Step 6: Carry propagation

Walk left-to-right, extract base-2^16 digit, carry the rest:
```
i=0: val = 131070 + 0    = 131070  → digit = 131070 & 0xFFFF = 0xFFFE
                                      carry = 131070 >> 16   = 1
i=1: val = 131070 + 1    = 131071  → digit = 131071 & 0xFFFF = 0xFFFF
                                      carry = 131071 >> 16   = 1
i=2: val = 2 + 1         = 3       → digit = 3
                                      carry = 0
i=3: val = 0             = 0       → digit = 0
                                      carry = 0
```

### Step 7: Limb assembly

Pack pairs of base-2^16 digits back into uint32 limbs:
```
result[0] = digit[0] | (digit[1] << 16) = 0xFFFE | (0xFFFF << 16) = 0xFFFFFFFE
result[1] = digit[2] | (digit[3] << 16) = 0x0003 | (0x0000 << 16) = 0x00000003
```

Result: `{0xFFFFFFFE, 0x00000003}` = 0x3FFFFFFFE = 17,179,869,182. ✓

## Baseline performance

Measured on NVIDIA L40S (Ada Lovelace, sm_89, 46GB).
Batch of 10 multiplications per hyperfine run, binary I/O mode was
not available in the baseline — results use hex I/O.

| n (limbs) | per-multiply (ms) |
|-----------|-------------------|
| 1,024     | 24.0              |
| 4,096     | 27.7              |
| 16,384    | 41.6              |
| 65,536    | 98.9              |
| 262,144   | 325.8             |
| 1,048,576 | 1,221.6           |
| 2,097,152 | 2,378.5           |

### Where the time goes

From nsys profiling:

- **GPU kernel time: ~200µs** (0.01% of wall-clock)
  - butterfly: 86% of kernel time
  - bit_reverse_permute: 8%
  - pointwise_mul: 3%
  - scale_mod: 3%

- **cudaMalloc: 98.8% of CUDA API time** (~100ms for context init)

- **Host-side work: ~99.9% of wall-clock**
  - Hex parsing (strtoul per 8 chars)
  - Digit splitting (CPU loop)
  - CRT (4M modular divisions per multiply)
  - Carry propagation (CPU loop)
  - Hex formatting (snprintf per limb)

### Key observation

The GPU is almost idle. Compute throughput: 0.4% of peak. Memory
throughput: 0.1% of peak. The bottleneck is entirely host-side — data
conversion, modular arithmetic, and I/O formatting.

This means GPU kernel optimizations (shared memory, coalescing, warp
shuffles) would have negligible impact. The optimization path must
first address the host-side overhead before GPU improvements become
visible.

## Kernel launch pattern

Per multiply at 2M limbs (NTT length m = 8M):

| Kernel | Launches | Purpose |
|--------|----------|---------|
| bit_reverse_permute | 6 | one per NTT (3 primes × 2 directions) |
| butterfly | 138 | 23 stages × 6 NTTs |
| pointwise_mul | 3 | one per prime |
| scale_mod | 3 | inverse NTT normalization |
| **Total** | **150** | |

Each kernel launch: ~5µs overhead. Total launch overhead: ~750µs.
Actual kernel compute: ~200µs. The kernels are extremely lightweight
individually — the parallelism is across many short launches rather
than few long ones.

## Code walkthrough

### `include/bigmul/bigmul.cuh`

**`check_cuda(err)`** — wraps every CUDA API call. Uses C++20
`std::source_location` to print the file and line where the error
occurred, then exits. `[[gnu::always_inline]]` ensures no function
call overhead.

**`BLOCK_SIZE = 256`** — compile-time constant for all kernel launches.
256 threads per block is a common choice — enough to hide latency,
not so many that register pressure becomes an issue.

**`bigmul(a, b, result, n)`** — the public API. Takes two n-limb
arrays on the host, writes 2n-limb result. All GPU work is internal.

### `include/bigmul/ntt.cuh`

**`NttPrime`** — struct with two fields: the prime `p` and its
primitive root `g`. The root is needed to compute roots of unity
for the NTT.

**`NTT_P1, NTT_P2, NTT_P3`** — the three primes as `constexpr`
values. Each is of the form `k × 2^m + 1`, which guarantees that
primitive 2^m-th roots of unity exist mod p.

**`mod_pow_host`** — modular exponentiation on the CPU. Used to
compute roots of unity (`g^((p-1)/n) mod p`) and modular inverses
via Fermat's little theorem (`a^(p-2) mod p = a^-1 mod p`).

**`ntt_forward / ntt_inverse / ntt_pointwise_mul`** — the NTT
operations. All take device pointers — data must already be on the
GPU.

### `src/bigmul/ntt.cu`

#### Device helper functions

**`mod_mul(a, b, p)`** — `(uint64_t)a * b % p`. The cast to 64-bit
prevents overflow since a,b < p < 2^30.

**`mod_add(a, b, p)`** — addition with conditional subtraction of p.
No branching in practice (compiler emits a conditional move).

**`mod_sub(a, b, p)`** — subtraction with conditional addition of p.

**`mod_pow(base, exp, p)`** — device-side modular exponentiation.
Binary exponentiation: square-and-multiply, log2(exp) iterations.
Used by `compute_twiddles` so each thread computes its own twiddle
independently.

#### GPU kernels

**`compute_twiddles(tw, w, p, n)`** — each thread i computes
`tw[i] = w^i mod p` using `mod_pow`. Embarrassingly parallel — no
thread depends on another. This replaced a sequential CPU loop that
was a major bottleneck.

**`bit_reverse_permute(data, n, log_n)`** — rearranges data into
bit-reversed order. Thread i computes `rev = bit_reverse(i)` by
reversing the bottom `log_n` bits. Then swaps `data[i]` and
`data[rev]` if `i < rev` (to avoid double-swapping).

**`butterfly(data, twiddles, stage, n, p)`** — one butterfly stage
of the NTT. Thread k maps to a pair (i, j) where:
- `half = 2^stage` (the stride for this stage)
- `group = k / half`, `pos = k % half`
- `i = group × 2 × half + pos`, `j = i + half`
- twiddle = `twiddles[step × pos]` where `step = n / (2 × half)`

The butterfly operation:
```
u = data[i]
v = data[j] × twiddle mod p
data[i] = u + v mod p
data[j] = u - v mod p
```

**`scale_mod(data, n, factor, p)`** — multiplies every element by
`factor mod p`. Used after inverse NTT to divide by n (multiply by
`n^-1 mod p`).

**`pointwise_mul(out, a, b, n, p)`** — element-wise modular multiply.
After forward NTT, convolution becomes pointwise multiplication in
the frequency domain.

#### Host functions

**`ntt_core(d_data, n, w, p)`** — the NTT engine. Computes twiddle
factors on GPU, does bit-reverse permutation, then runs `log2(n)`
butterfly stages. Uses a static device buffer for twiddles (pooled
across calls).

**`ntt_forward(d_data, n, prime)`** — computes the primitive n-th
root of unity `w = g^((p-1)/n) mod p`, then calls `ntt_core`.

**`ntt_inverse(d_data, n, prime)`** — same as forward but with the
inverse root `w^-1 = w^(p-2) mod p`. After `ntt_core`, scales all
elements by `n^-1 mod p` to normalize.

**`ntt_pointwise_mul(d_out, d_a, d_b, n, p)`** — launches the
pointwise multiply kernel.

### `src/bigmul/bigmul.cu`

#### GPU kernels

**`digit_split(limbs, digits, n)`** — each thread splits one uint32
limb into two base-2^16 digits. `digits[2i] = limbs[i] & 0xFFFF`,
`digits[2i+1] = limbs[i] >> 16`. Runs on GPU to avoid a host loop
and an extra H2D transfer.

**`crt_kernel(r1, r2, r3, conv, m, ...)`** — Chinese Remainder
Theorem via Garner's algorithm. Each thread reconstructs the true
convolution value at one position from the three mod-p residues.
The intermediate computation overflows uint64 but the final result
fits (< 2^54), so unsigned wraparound gives the correct answer.

**`carry_and_assemble(conv, result, n, m)`** — serial kernel (one
thread) that walks the convolution values, extracts base-2^16 digits,
propagates carries, and packs pairs of digits back into uint32 limbs.
Sequential because each carry depends on the previous position.

#### The main function: `bigmul(a, b, result, n)`

1. Compute NTT length: `m = next_power_of_2(4n)`
2. Allocate device buffers (pooled — reused across calls)
3. Upload raw limbs to GPU (one H2D copy per input)
4. `digit_split` — split uint32 → base-2^16 on GPU
5. For each of 3 primes:
   - D2D copy digits to working buffers (NTT is in-place)
   - Forward NTT on both inputs
   - Pointwise multiply
   - Inverse NTT on the product
6. CRT kernel — combine 3 residues into true values
7. Carry + assemble kernel — propagate carries, build result limbs
8. Download result (one D2H copy)

### `src/multiply/main.cu`

The CLI binary. Three modes:

**Hex args** — `./multiply FF 2` prints `1FE`. Parses hex strings to
limbs, calls `bigmul`, formats result back to hex.

**Batch mode** — `--batch` reads pairs of hex lines from stdin. Each
pair is an independent multiplication. Used for benchmarking with
amortized process startup.

**Binary mode** — `--binary` reads `[n:u32][a:u32×n][b:u32×n]` from
stdin, writes `[result:u32×2n]` to stdout. No parsing overhead. Used
for performance benchmarking.

**`hex_to_limbs`** — parses a hex string right-to-left in 8-char
chunks, each becoming one uint32 limb via `strtoul`.

**`limbs_to_hex`** — formats limbs to hex. Top limb gets minimal
digits (no leading zeros), rest get zero-padded to 8 chars.

### `script/test`

Correctness verification. Runs multiply on 14 hardcoded test cases
plus 5 random ones, pipes all results to a single Perl process that
checks against `Math::BigInt`.

### `script/bench`

Benchmark driver. Generates binary input pairs at sizes from 1K to
2M limbs, runs hyperfine (warmup + 5 runs per size), then nsys and
ncu profiling. Outputs `data/timings.csv` and profiling reports.

### `script/report`

Extracts metrics from nsys (SQLite) and ncu (CSV) into a single
`result/report.xlsx` with sheets: Summary, Timings, Kernels, Memcpy,
CUDA API, NCU Metrics.

### `script/profile`

Orchestrator. Rsyncs code to the cluster, runs `script/bench` via
srun on a grace node, fetches results back locally.

### `script/plot`

Generates 4 PNG charts from `report.xlsx`: timing curve, kernel
breakdown, butterfly profile, top stall reasons.
