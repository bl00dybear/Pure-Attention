#pragma once

#include <memory>
#include <vector>
#include <cuda_runtime.h>

namespace core {

    struct CudaDeallocator {
        void operator()(void* ptr) const {
            if (ptr) {
                cudaFree(ptr);
            }
        }
    };

    using float32_t = float;

    class Tensor : public std::enable_shared_from_this<Tensor> {
    private:
        std::unique_ptr<float32_t, CudaDeallocator> data_ptr;
        std::unique_ptr<float32_t, CudaDeallocator> gradient_ptr;

        std::vector<uint32_t> shape;
        size_t size;
        bool has_gradient; 

    public:
        Tensor(const std::vector<uint32_t>& shape, bool has_gardient = false);
        ~Tensor() = default;

        static std::shared_ptr<Tensor> zeros(); 
        static std::shared_ptr<Tensor> rand_normal(); 
        void to_device(const std::vector<float32_t>& host_data);
        std::vector<float32_t> to_host();

        std::vector<uint32_t> get_shape() const ;
        float *get_data_ptr() const ;
    };
};