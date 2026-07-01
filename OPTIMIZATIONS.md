# Optimization Story

Same NTT algorithm throughout. Each chapter: profile → discover bottleneck → fix → measure.

## Baseline

- L40S (Ada, sm_89), 2M limbs = 1428ms
- 195ms floor from cudaMalloc (even at 1K limbs)
- Butterfly: 86% of kernel time
- Compute throughput: 0.4%, memory throughput: 0.1% — GPU almost idle
- 141 kernel launches per multiply
- Top stalls: imc_miss, long_scoreboard

## Chapter 1: Memory allocation overhead

**Discovery:** nsys shows cudaMalloc is 98.8% of CUDA API time. First
cudaMalloc lazily initializes the CUDA context (~100ms). Subsequent
allocations add ~5ms each, 12 calls per multiply.

**Fix:** Pre-allocate device buffers once, reuse across calls. Move
cudaMalloc out of bigmul() into an init function or allocate on first
call and cache.

**Expected result:** Wall-clock floor drops from 195ms to ~30ms. Small
input performance improves dramatically. Large input overhead fraction
shrinks.

## Chapter 2: Kernel launch overhead

**Discovery:** After removing allocation overhead, profile reveals 141
kernel launches per multiply (log2(N) butterfly stages × 6 NTTs + bit
reverse + pointwise + scale). Each launch has ~5µs overhead.

**Fix:** Fuse multiple butterfly stages into a single kernel using shared
memory. Process several stages while data fits in shared memory before
writing back to global memory.

**Expected result:** Launch count drops from 141 to ~30-40. Compute
utilization increases.

## Chapter 3: Memory coalescing

**Discovery:** Butterfly kernel shows imc_miss and long_scoreboard as top
stalls. Early butterfly stages have strided global memory access — thread
k accesses data[k] and data[k + half], where half doubles each stage.

**Fix:** Use shared memory to stage data into a coalesced layout before
butterfly operations. Twiddle factors also loaded into shared memory
to reduce redundant global reads.

**Expected result:** Memory throughput increases, stalls shift from
memory-bound to compute-bound.

## Chapter 4: Warp-level primitives

**Discovery:** After shared memory optimization, small butterfly stages
(stride ≤ 32) still use shared memory unnecessarily — data is within a
single warp.

**Fix:** Use __shfl_xor_sync for butterfly operations when stride ≤ 32.
No shared memory, no synchronization, no bank conflicts.

**Expected result:** Reduced shared memory pressure, faster small stages.

## Chapter 5: Stream concurrency

**Discovery:** Profile shows 3 NTT passes (one per prime) run
sequentially. GPU utilization drops to zero between passes.

**Fix:** Run each prime's NTT on a separate CUDA stream. Forward NTT,
pointwise multiply, and inverse NTT for all 3 primes overlap.

**Expected result:** ~3x improvement in NTT phase if memory-bandwidth
allows. Realistically 1.5-2x due to shared bandwidth.

## Chapter 6: GPU-side CRT and carry propagation

**Discovery:** After GPU optimizations, host-side CRT + carry
propagation becomes the bottleneck — sequential O(n) work with a
cudaMemcpy sync point before and after.

**Fix:** Move CRT (Garner's algorithm) and carry propagation to GPU
kernels. Eliminate host-device round trip.

**Expected result:** Remove CPU bottleneck, pipeline runs entirely on GPU
except for initial upload and final download.
