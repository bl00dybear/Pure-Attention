#include <backend/Launchers.cuh>
#include <backend/Kernels.cuh>
#include <vector>


void launch_concat_backward(
    std::vector<float*>& input_grads,
    float* output_grad,
    uint32_t num_splits, 
    uint32_t inner_size, 
    uint32_t split_size, 
    uint32_t total_elements, 
    cudaStream_t stream)
{
    float** d_in_ptrs;
    cudaMallocAsync(&d_in_ptrs, num_splits * sizeof(float*), stream);
    cudaMemcpyAsync(d_in_ptrs, input_grads.data(), num_splits * sizeof(float*), cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    concat_last_dim_kernel<<<blocks, threads, 0, stream>>>(
        d_in_ptrs,
        output_grad,
        num_splits,
        inner_size,
        split_size,
        total_elements
    );

    cudaFreeAsync(d_in_ptrs, stream);
}
