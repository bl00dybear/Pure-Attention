// headers
#include <core/Tensor.h>
#include <core/Context.h>

// libs
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace core {
    Tensor::Tensor(const std::vector<uint32_t>& shape, bool has_gradient) : 
    shape(shape), 
    has_gradient(has_gradient) 
    {
        size = 1;
        for (int s : shape) {
            size *= s;
        }

        float32_t* raw_ptr_data = nullptr;
        size_t bytes = size * sizeof(float32_t);

        cudaError_t err = cudaMalloc((void**)&raw_ptr_data, bytes);

        if (err != cudaSuccess) {
            std::string error_msg = "CUDA Error: " + std::string(cudaGetErrorString(err));
            throw std::runtime_error(error_msg);
        }

        data_ptr.reset(raw_ptr_data);

        if (has_gradient) {
            float32_t* raw_ptr_grad = nullptr;
            size_t bytes = size * sizeof(float32_t);
    
            cudaError_t err = cudaMalloc((void**)&raw_ptr_grad, bytes);
    
            if (err != cudaSuccess) {
                std::string error_msg = "CUDA Error: " + std::string(cudaGetErrorString(err));
                throw std::runtime_error(error_msg);
            }
    
            gradient_ptr.reset(raw_ptr_grad);
        }
    }

    void Tensor::to_device(const std::vector<float32_t>& host_data) {
        if (host_data.size() != size) {
            throw std::length_error("Size mismatch: CPU array has different size to GPU tensor.");
        }

        cudaError_t err = cudaMemcpyAsync(data_ptr.get(), host_data.data(), size * sizeof(float32_t), cudaMemcpyHostToDevice, CudaContext::getStream());
        
        if (err != cudaSuccess) {
            throw std::runtime_error("To GPU error (Memcpy H2D)");
        }
    }

    std::vector<float32_t> Tensor::to_host() {
        std::vector<float32_t> host_result(size);

        cudaError_t err = cudaMemcpyAsync(host_result.data(), data_ptr.get(), size * sizeof(float32_t), cudaMemcpyDeviceToHost, CudaContext::getStream());

        if (err != cudaSuccess) {
            throw std::runtime_error("To CPU error (Memcpy D2H)");
        }
        
        return host_result;
    }

    std::vector<uint32_t> Tensor::get_shape() const {
        return shape;
    };

    float *Tensor::get_data_ptr() const {
        return data_ptr.get();
    };
};


// nvcc -std=c++20 -Iinclude -o forward \
//   main.cpp \
//   source/layers/Linear.cpp \
//   source/layers/ReLU.cpp \
//   source/core/Functional.cpp \
//   source/core/Tensor.cpp \
//   source/backend/Launchers.cu \
//   source/backend/Kernels.cu \
//   -lcurand -lcudart