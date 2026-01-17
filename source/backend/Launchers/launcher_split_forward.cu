#include <backend/Kernels.cuh>
#include <backend/Launchers.cuh>


void launch_split_forward(
    const float32_t* input, 
    std::vector<float32_t*>& output_list, 
    uint32_t num_splits,
    uint32_t inner_size,
    uint32_t split_size,
    uint32_t total_elements,
    cudaStream_t stream
){
    float32_t** d_out_ptrs;
    cudaMallocAsync(&d_out_ptrs, num_splits * sizeof(float32_t*), stream);

    cudaMemcpyAsync(d_out_ptrs, output_list.data(), num_splits * sizeof(float32_t*), cudaMemcpyHostToDevice, stream);
    
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    split_last_dim_kernel<<<blocks, threads, 0, stream>>>(
        input,
        d_out_ptrs,
        num_splits,
        inner_size,
        split_size,
        total_elements
    );
    
    cudaFreeAsync(d_out_ptrs, stream);
}
