# Optimization Roadmap

Baseline profiled on NVIDIA L40S (grace1, sm_89). Same NTT algorithm throughout — computational optimizations only.

## Low effort, high impact

### 1. Memory pool
cudaMalloc is 98.8% of CUDA API time. Allocate device buffers once, reuse across calls.

### 2. Persistent twiddle factors
Twiddle arrays allocated, computed, and freed every NTT call. Compute once at init, keep on device.

### 3. CUDA streams
Three prime NTTs run sequentially. Overlap them on separate streams for free concurrency.

## Medium effort, medium impact

### 4. Fused butterfly stages
One kernel launch per stage — ~120 launches per multiply. Fuse multiple stages into one kernel using shared memory while data fits.

### 5. Merge bit-reverse with first butterfly
bit_reverse_permute is 7.5% of GPU time. Fold into the first butterfly kernel.

### 6. GPU-side CRT and carry propagation
Currently sequential on CPU. Move to GPU to eliminate host-device sync point.

## High effort

### 7. Warp shuffles for small stages
Butterfly stages where stride <= 32 can use __shfl_xor instead of global memory.

### 8. Coalesced access pattern
Early butterfly stages have strided global memory access. Reorder data layout or use shared memory to coalesce.
