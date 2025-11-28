#include <backend/Kernels.cuh>

__global__ void matmul_kernel_tiled(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int global_row = blockIdx.y * TILE_WIDTH + ty;
    int global_col = blockIdx.x * TILE_WIDTH + tx;

    int tile_num_reduction=(N+TILE_WIDTH-1)/TILE_WIDTH;
    float val = 0.0f;

    for(int m = 0; m < tile_num_reduction; m += 1) {
        int global_read_row_A = global_row;
        int global_read_col_A = m * TILE_WIDTH + tx;

        int global_read_row_B = m * TILE_WIDTH + ty;
        int global_read_col_B = global_col;
        
        if(global_read_col_A < N && global_read_row_A < M) {
            s_A[ty][tx] = A[global_read_row_A * N + global_read_col_A];
        }
        else {
            s_A[ty][tx] = 0.0f;
        }

        if(global_read_col_B < K && global_read_row_B < N) {
            s_B[ty][tx] = B[global_read_row_B * K + global_read_col_B];
        }
        else {
            s_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            val += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    } 

    if (global_row < M && global_col < K) {
        C[global_row * K + global_col] = val;
    }
}

__global__ void matadd_kernel_tiled(const float *A, const float *X, float *B, const int M, const int N) {
    int global_col = blockIdx.x * blockDim.x + threadIdx.x;
    int global_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (global_row < M && global_col < N) {
        int index = global_row * N + global_col;
        B[index] = A[index] + X[global_col];
    }
}

__global__ void populate_normal(float *A, int M, int N, unsigned long long seed) {
    const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int total = M * N;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid0, 0, &state);

    for (int i = tid0; i < total; i += stride) {
        A[i] = curand_normal(&state);
    }
}

