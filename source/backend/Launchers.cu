#include <backend/Launchers.h>
#include <backend/Kernels.cuh>

void launch_matmul_tiled(float *A, float *B, float *C, int M, int N, int K, cudaStream_t stream) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matmul_kernel_tiled<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

void launch_matadd_tiled(float *A, float *X, float *B, int M, int N, cudaStream_t stream) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matadd_kernel_tiled<<<grid, block, 0, stream>>>(A, X, B, M, N);
}

void launch_zero_population(float *A, int M, int N, cudaStream_t stream){
    size_t total = static_cast<size_t>(M) * static_cast<size_t>(N);
    cudaMemsetAsync(A, 0, total * sizeof(float), stream);
}

void launch_normal_population(float *A, int M, int N, cudaStream_t stream){
    size_t total = static_cast<size_t>(M) * static_cast<size_t>(N);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, device);

    int threads = std::min(256, device_props.maxThreadsPerBlock);
    size_t blocks = (total + threads - 1) / threads;

    int activeBlocksPerSm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSm,
        populate_normal,    
        threads,
        0                    
    );

    size_t max_blocks = static_cast<size_t>(device_props.multiProcessorCount) * static_cast<size_t>(std::max(1, activeBlocksPerSm));
    if (blocks > max_blocks) blocks = max_blocks;

    unsigned long long seed = static_cast<unsigned long long>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
    );

    seed=42;
    populate_normal<<<blocks, threads, 0, stream>>>(A, M, N, seed);
}

void launch_ReLU_tiled(float *In, float *Out, int N, cudaStream_t stream) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + block.x - 1) / block.x);

    ReLU_kernel_tiled<<<grid, block, 0, stream>>>(In, Out, N);
}