#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

void bigmul(const uint32_t* a, const uint32_t* b, uint32_t* result, int n);
