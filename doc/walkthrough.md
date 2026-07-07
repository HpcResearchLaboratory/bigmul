# Code Walkthrough — Baseline (v0)

## Entry point: `main()` in `src/multiply/main.cu`

```
./multiply FF 2
```

**Line 84-88: mode dispatch.** Checks the first argument. `--batch` reads
hex pairs from stdin. Otherwise falls through to hex-args mode.

**Line 90-97: collect arguments.** If ≥3 args, take them from argv. Otherwise
read lines from stdin. Either way, `args` is a vector of hex strings.

**Line 105: `run_chain(args)`.** Multiplies all args together: A×B×C = ((A×B)×C).

### `run_chain`

**Line 69: `hex_to_limbs(args[0])`.** Converts the first hex string to a
uint32 limb array. Walks the string right-to-left in 8-char chunks. Each
chunk → one uint32 via `strtoul`. Example: `"1FFFFFFFF"` → `{0xFFFFFFFF, 0x1}`.

**Line 70-78: multiplication loop.** For each subsequent argument:
- Parse it to limbs
- Pad both to the same length
- Allocate a result array of double length
- Call **`bigmul()`** — this is where all GPU work happens
- Strip trailing zero limbs from result
- Result becomes the accumulator for the next iteration

**Line 80: `limbs_to_hex`.** Converts the final result back to hex. Top limb
gets no leading zeros, rest are zero-padded to 8 hex chars.

---

## The core: `bigmul()` in `src/bigmul/bigmul.cu`

Called with two host arrays of n uint32 limbs. Returns 2n limbs.

### Phase 1: Digit splitting (CPU)

**Line 7-8: compute NTT length.**
```cpp
int n_digits = 2 * n;    // each uint32 → 2 base-2^16 digits
int m = 1;
while (m < 2 * n_digits) m <<= 1;  // next power of 2 ≥ 4n
```
For n=1M limbs: n_digits=2M, m=8M.

**Line 11-17: split limbs into base-2^16 digits.**
```cpp
std::vector<uint32_t> da(m, 0), db(m, 0);   // heap alloc, zero-filled
for (int i = 0; i < n; i++) {
    da[2*i]   = a[i] & 0xFFFF;    // low 16 bits
    da[2*i+1] = a[i] >> 16;       // high 16 bits
}
```
This is a CPU loop over n elements. At n=2M that's 2M iterations plus
two 32MB heap allocations. All on the host.

### Phase 2: NTT multiply (GPU, 3 passes)

**Line 19-20: prime setup.**
```cpp
const NttPrime primes[] = {NTT_P1, NTT_P2, NTT_P3};
std::vector<uint32_t> res[3];
```
Three NTT-friendly primes. `res[3]` will hold the three convolution
results on the host.

**Line 22-25: device memory allocation.**
```cpp
uint32_t *d_a, *d_b, *d_c;
size_t bytes = m * sizeof(uint32_t);
check_cuda(cudaMalloc(&d_a, bytes));   // ~32MB at 2M limbs
check_cuda(cudaMalloc(&d_b, bytes));
check_cuda(cudaMalloc(&d_c, bytes));
```
Three device buffers. The first `cudaMalloc` also lazily initializes the
CUDA context (~100ms). These are allocated and freed EVERY call to bigmul.

**Line 27-37: the 3-prime loop.**
```cpp
for (int pi = 0; pi < 3; pi++) {
    cudaMemcpy(d_a, da.data(), bytes, H2D);  // upload digits
    cudaMemcpy(d_b, db.data(), bytes, H2D);  // same data each time!

    ntt_forward(d_a, m, primes[pi]);          // → frequency domain
    ntt_forward(d_b, m, primes[pi]);
    ntt_pointwise_mul(d_c, d_a, d_b, m, ...); // element-wise mod multiply
    ntt_inverse(d_c, m, primes[pi]);           // → time domain

    res[pi].resize(m);                         // alloc host buffer
    cudaMemcpy(res[pi].data(), d_c, bytes, D2H); // download result
}
```
Each iteration: 2 uploads + NTT work + 1 download. The same digit arrays
are re-uploaded 3 times (wasteful). The NTT is in-place so d_a and d_b
are destroyed — must re-upload for each prime.

**Line 39-41: free device memory.**
```cpp
cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
```
Freed every call. Next call re-allocates.

### Phase 3: CRT — Garner's algorithm (CPU)

**Line 43-46: precompute modular inverses.**
```cpp
uint32_t p1_inv_p2 = mod_pow_host(p1, p2 - 2, p2);  // p1^(-1) mod p2
uint32_t p1_inv_p3 = mod_pow_host(p1, p3 - 2, p3);
uint32_t p2_inv_p3 = mod_pow_host(p2, p3 - 2, p3);
```
Fermat's little theorem: a^(p-2) mod p = a^(-1) mod p. Three modular
exponentiations on CPU. Fast (~1µs each).

**Line 48-70: CRT + carry propagation loop.**
```cpp
for (int i = 0; i < out_digits; i++) {
```
Loops over 4n positions (4M iterations at n=1M). Each iteration:

1. **Read three residues** — `r1 = res[0][i]`, `r2 = res[1][i]`, `r3 = res[2][i]`

2. **Garner's algorithm** — reconstruct the true convolution value from
   three mod-p residues:
   ```
   a1 = r1
   a2 = (r2 - a1) × p1^(-1) mod p2
   a3 = ((r3 - a1) × p1^(-1) - a2) × p2^(-1) mod p3
   x  = a1 + a2×p1 + a3×p1×p2
   ```
   Uses `__uint128_t` for the final reconstruction (product can be ~2^88).

3. **Carry propagation** — extract base-2^16 digit from x + carry:
   ```
   digit = (x + carry) & 0xFFFF
   carry = (x + carry) >> 16
   ```

4. **Limb assembly** — pack pairs of digits into uint32:
   ```
   if even: result[i/2]  = digit
   if odd:  result[i/2] |= digit << 16
   ```

This entire loop is sequential on CPU. Each iteration does multiple
64-bit divisions (the `%` operations in Garner's) and one 128-bit
multiply. At n=2M this takes ~150ms.

---

## Inside the NTT: `ntt.cu`

### `ntt_forward(d_data, n, prime)`

**Line 97-99:** Computes the primitive n-th root of unity:
```cpp
uint32_t w = mod_pow_host(prime.g, (prime.p - 1) / n, prime.p);
```
`g` is the primitive root of the prime. `g^((p-1)/n)` gives a value
whose n-th power ≡ 1 mod p. This `w` is the "twiddle base" for all
butterfly operations.

Calls `ntt_core(d_data, n, w, prime.p)`.

### `ntt_core(d_data, n, w, p)`

**Line 80-85: twiddle precomputation (CPU).**
```cpp
std::vector<uint32_t> tw(n);       // 32MB heap alloc at n=8M
tw[0] = 1;
for (int i = 1; i < n; i++)
    tw[i] = (uint64_t)tw[i-1] * w % p;  // sequential, 8M iterations
```
Computes w^0, w^1, w^2, ..., w^(n-1) mod p. Sequential because each
value depends on the previous. At n=8M this takes ~24ms per call,
called 6 times per multiply = ~144ms.

**Line 87-88: upload twiddles.**
```cpp
cudaMalloc(&d_tw, n * sizeof(uint32_t));     // 32MB alloc
cudaMemcpy(d_tw, tw.data(), ..., H2D);       // 32MB upload
```
Allocated and freed per NTT call (6 times per multiply).

**Line 91-92: bit-reverse permutation.**
```cpp
bit_reverse_permute<<<(n+B-1)/B, B>>>(d_data, n, log_n);
```
Launches one kernel: n threads, each computes bit_reverse(i) and
swaps data[i] with data[rev] if i < rev. This rearranges the array
so the butterfly stages produce output in natural order.

**Line 94-97: butterfly stages.**
```cpp
for (int stage = 0; stage < log_n; stage++) {
    butterfly<<<(n/2+B-1)/B, B>>>(d_data, d_tw, stage, n, p);
}
```
log2(n) kernel launches (23 at n=8M). Each launch: n/2 threads.
Each thread does one butterfly:
- Read data[i] and data[j] where j = i + 2^stage
- Multiply data[j] by the twiddle factor
- Write data[i] = u + v, data[j] = u - v (all mod p)

**Line 99-100: sync and free twiddles.**
```cpp
cudaDeviceSynchronize();
cudaFree(d_tw);
```

### `ntt_inverse`

Same as forward but:
- Uses w^(-1) instead of w (computed via Fermat's)
- After ntt_core, multiplies every element by n^(-1) mod p via `scale_mod` kernel

### `butterfly` kernel

The heart of the NTT. Thread k maps to one butterfly pair:
```
half = 2^stage                    // stride for this stage
group = k / half                  // which group of butterflies
pos = k % half                   // position within group
i = group × 2 × half + pos       // first element
j = i + half                     // second element (stride away)
tw = twiddles[step × pos]        // twiddle factor for this position

u = data[i]
v = data[j] × tw mod p           // modular multiply
data[i] = (u + v) mod p
data[j] = (u - v) mod p
```

### Modular arithmetic helpers

- `mod_mul(a, b, p)` → `(uint64_t)a * b % p` — cast to 64-bit to avoid overflow
- `mod_add(a, b, p)` → `a + b`, subtract p if ≥ p
- `mod_sub(a, b, p)` → `a - b`, add p if negative

---

## Summary of baseline data flow

```
Host: hex string
  → hex_to_limbs (CPU, char-by-char strtoul)
  → uint32 limbs

Host: digit splitting (CPU loop, O(n))
  → base-2^16 digits in std::vector

Host→GPU: cudaMemcpy H2D (×6, same data re-uploaded per prime)

GPU: for each of 3 primes:
  CPU: twiddle precomputation (sequential loop, 8M iterations)
  Host→GPU: twiddle upload (cudaMalloc + cudaMemcpy each time)
  GPU: bit_reverse_permute (1 kernel)
  GPU: butterfly × 23 stages (23 kernels)
  GPU: pointwise_mul (1 kernel)
  GPU: bit_reverse_permute (1 kernel)
  GPU: butterfly × 23 stages (23 kernels)
  GPU: scale_mod (1 kernel)
  GPU→Host: cudaMemcpy D2H result

Host: CRT + carry propagation (CPU loop, 4M iterations, modular divisions)
  → uint32 result limbs

Host: limbs_to_hex (CPU, snprintf per limb)
  → hex string → stdout
```

**Total kernel launches per multiply: ~150**
**Total cudaMalloc/Free: 9 in bigmul + 6 in ntt_core = 15 pairs**
**Host-side bottlenecks: hex I/O, digit splitting, twiddle computation, CRT**

---

# What changed in each optimization

## v1-gpu-crt: Move CRT to GPU

**What we noticed:** nsys showed GPU kernels took ~200µs but wall-clock
was ~2400ms. 99.9% was host-side. The CRT loop does 4M iterations of
modular divisions — expensive on CPU.

**What changed in `bigmul.cu`:**
- Removed the host CRT loop (lines 48-70)
- Added `crt_kernel` — a GPU kernel where each thread handles one
  position. Same Garner's algorithm, but parallel across all positions.
- Need 3 separate d_c buffers (d_c[3]) since results must coexist for CRT
- CRT output is a `uint64_t` conv array on device
- Carry propagation stays on CPU — download conv, walk sequentially
- Eliminated 3 D2H copies of per-prime results, replaced with 1 D2H of
  conv array

**Result:** ~4% faster at large sizes.

## v2-io: Binary I/O, GPU digit split and carry

**What we noticed:** Even after GPU CRT, 99% was still host-side. The
remaining bottlenecks: hex parsing (~200ms), hex formatting (~200ms),
digit splitting loop (~50ms), carry propagation on CPU (~50ms).

**What changed:**
- `main.cu`: Added `--binary` mode. Reads `[n:u32][a:u32×n][b:u32×n]`
  from stdin, writes raw result. No hex parsing or formatting.
- `bigmul.cu`: Removed the CPU digit splitting loop. Added `digit_split`
  GPU kernel — each thread splits one limb into two base-2^16 digits.
  Upload raw limbs to GPU, split there.
- `bigmul.cu`: Added `carry_and_assemble` GPU kernel — single thread
  does carry propagation on GPU. Serial but avoids D2H of conv array.
- Upload raw limbs once, then D2D copy per prime instead of H2D 3×.
- `script/bench`: Generates binary inputs with python, uses `--binary`.

**Result:** 1.5-1.8× faster. Stddev dropped from ~5% to <0.1%.

## v3-pool: Memory pool

**What we noticed:** cudaMalloc was still being called 13 times per
multiply (12 in bigmul + 1 twiddle). With batch mode, we call bigmul
10 times per process — first call pays ~100ms, rest pay ~5ms each.

**What changed in `bigmul.cu`:**
- Device pointers changed from local to `static`.
- Added `pool_limb` and `pool_digit` tracking current capacity.
- On first call (or when size grows), allocate. Otherwise reuse.
- Removed `cudaFree` calls at end of bigmul.

**What changed in `ntt.cu`:**
- Same pattern for the twiddle buffer `d_tw`.

**Result:** ~1-3% faster. Statistically significant with binary mode.

## v4-gpu-twiddle: GPU twiddle computation

**What we noticed:** `ntt_core` computed twiddles in a sequential CPU
loop: 8M modular multiplications per NTT, 6 NTTs per multiply = ~144ms.
Plus 6 × 32MB std::vector allocations and H2D memcpy each call.

**What changed in `ntt.cu`:**
- Added `mod_pow` as a `__device__` function (same as host version).
- Added `compute_twiddles` kernel — each thread i computes `w^i mod p`
  independently via modular exponentiation. No sequential dependency.
- Removed the CPU loop, the std::vector, and the cudaMemcpy.
- Twiddles go straight to device memory.

**Result:** 23-27% faster. The biggest single optimization.

## v5-one-prime: Goldilocks prime

**What we noticed:** Running 3 NTT passes with CRT is 3× the GPU work
of a single pass. If we could use one prime large enough to hold the
convolution values, CRT is unnecessary.

**What changed:**
- `ntt.cuh`: Replaced three 32-bit primes with one 64-bit prime:
  `p = 0xFFFFFFFF00000001` (Goldilocks prime, 2^64 - 2^32 + 1).
  Supports NTT sizes up to 2^32. Primitive root: 7.
- All NTT types changed from `uint32_t` to `uint64_t`.
- `bigmul.cu`: Removed the 3-prime loop and CRT kernel. Single NTT pass.
  Only one d_c buffer instead of three.
- Modular arithmetic now uses 64-bit operands (128-bit intermediate for
  mod_mul).

**Result:** ~22% faster at 2M limbs. Simpler code, fewer allocations.

## v6-carry-paralelo: Parallel carry propagation

**What we noticed:** The serial `carry_and_assemble<<<1,1>>>` kernel was
the remaining bottleneck. One GPU thread doing 8M sequential iterations.

**What changed:**
- Added `carry.cu` and `carry.cuh` — a parallel carry propagation
  algorithm that splits the work across multiple threads.
- `bigmul.cu`: Replaced `carry_and_assemble<<<1,1>>>` with the parallel
  `carry_and_assemble()` host function from carry.cuh, which launches
  a multi-threaded carry kernel internally.

**Result:** 4.4× faster at 2M limbs. The single biggest optimization.

---

## Cumulative results at 2M limbs

```
v0-baseline        2,378 ms    1.0×
v1-gpu-crt         2,291 ms    1.04×
v2-io              1,525 ms    1.56×
v3-pool            1,518 ms    1.57×
v4-gpu-twiddle     1,164 ms    2.04×
v5-one-prime         904 ms    2.63×
v6-carry-paralelo    204 ms   11.66×
```
