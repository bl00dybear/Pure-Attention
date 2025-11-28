#pragma once

// libs
#include <cuda_runtime.h>
#include <chrono>


#define TILE_WIDTH 32

void launch_matmul_tiled(float *A, float *B, float *C, int M, int N, int K, cudaStream_t stream = 0);
void launch_matadd_tiled(float *A, float *X, float *B, int M, int N, cudaStream_t stream = 0);

void launch_zero_population(float *A, int M, int N, cudaStream_t stream = 0);
void launch_normal_population(float *A, int M, int N, cudaStream_t stream = 0);
