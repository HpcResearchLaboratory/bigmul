#pragma once

#include <cstdint>

void bigmul(const uint32_t* a, const uint32_t* b, uint32_t* result, int n);
void bigmul_cpu(const uint32_t* a, const uint32_t* b, uint32_t* result, int n);
