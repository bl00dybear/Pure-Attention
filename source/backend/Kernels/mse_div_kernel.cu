// headers
#include <backend/Kernels.cuh>

__global__ void mse_div_kernel(float32_t *loss_out,const uint32_t N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *loss_out /= static_cast<float32_t>(N);
    }
}
