// headers
#include <backend/Kernels.cuh>

__global__ void matadd_kernel_tiled(const float *A, const float *X, float *B, const int M, const int N) {
    const uint32_t global_col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t global_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (global_row < M && global_col < N) {
        const uint32_t index = global_row * N + global_col;
        B[index] = A[index] + X[global_col];
    }
} 
