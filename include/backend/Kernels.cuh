#pragma once

#define TILE_WIDTH 32

__global__ void matmul_kernel_tiled(const float *A, const float *B, float *C, int M, int N, int K);
__global__ void matadd_kernel_tiled(const float *A, const float *X, float *B, const int M, const int N);

