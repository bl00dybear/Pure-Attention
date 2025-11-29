#pragma once

// libs
#include <curand_kernel.h>

// macro
#define TILE_WIDTH 32

__global__ void matmul_kernel_tiled(const float *A, const float *B, float *C, int M, int N, int K);
__global__ void matadd_kernel_tiled(const float *A, const float *X, float *B, const int M, const int N);
__global__ void ReLU_kernel_tiled(const float *In, float *Out, int M, int N);
__global__ void populate_normal(float *A, int M, int N, unsigned long long seed);
