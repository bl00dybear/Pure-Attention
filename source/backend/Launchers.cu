#include <backend/Launchers.h>
#include <backend/Kernels.cuh>

void launch_matmul_tiled(float *A, float *B, float *C, int M, int N, int K, cudaStream_t stream = 0) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matmul_kernel_tiled<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

void launch_matadd_tiled(float *A, float *X, float *B, int M, int N, cudaStream_t stream = 0) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matadd_kernel_tiled<<<grid, block, 0, stream>>>(A, X, B, M, N);
}