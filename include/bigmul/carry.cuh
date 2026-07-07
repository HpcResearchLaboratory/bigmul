#pragma once

#include <cstdint>

// Carry-normalization algorithms adapted from crazynds/MillerRabin-GPU
// (src/ops/carry/carry_norm.cu), simplified for a single big-integer
// multiplication (no candidate batching) in base 2^16.
//
// Select the algorithm by defining CARRY_NORM_ALG before including this
// header (e.g. via -DCARRY_NORM_ALG=CARRY_ALG_MULTI_TILE), or by editing
// the default below.

#define CARRY_ALG_SEQUENTIAL 0
#define CARRY_ALG_SINGLE_TILE 1
#define CARRY_ALG_MULTI_TILE 2
#define CARRY_ALG_PREFIX_SCAN 3

#ifndef CARRY_NORM_ALG
#define CARRY_NORM_ALG CARRY_ALG_PREFIX_SCAN
#endif

// Tile size used by SINGLE_TILE / MULTI_TILE / PREFIX_SCAN. Must be a
// multiple of 32.
#ifndef CARRY_TILE
#define CARRY_TILE 32
#endif

// Normalizes m raw base-2^16 coefficients (arbitrary magnitude, as produced
// by the NTT convolution) held in d_conv, carry propagated directly into
// 2*n packed 32-bit result limbs written to d_result. Each thread/work-item
// owns one 32-bit limb (its two constituent base-2^16 coefficients), so no
// intermediate digit array or separate packing pass is needed.
//
// d_scratch is caller-owned device memory reused as scratch space (only
// CARRY_ALG_MULTI_TILE needs it, for its per-tile carry array); it must
// have at least m uint64_t elements available and its contents are not
// preserved. Pass a buffer that is no longer needed at this point (e.g.
// one of the NTT input buffers) to avoid an extra allocation here.
auto carry_and_assemble(const uint64_t* d_conv, uint32_t* d_result, int n, int m,
                        uint64_t* d_scratch) -> void;

// Batched variant: d_conv holds `batch` contiguous blocks of m raw
// coefficients, d_result `batch` contiguous blocks of 2*n packed limbs.
// d_scratch must have at least batch * (n_tiles+1) capacity in terms of
// uint64_t elements, where n_tiles = ceil(2*n / CARRY_TILE); passing a
// batch*m-sized buffer (as bigmul_batch does) is always large enough.
auto carry_and_assemble_batch(const uint64_t* d_conv, uint32_t* d_result, int n, int m, int batch,
                              uint64_t* d_scratch) -> void;
