#include "core/Tensor.h"
#include "backend/Kernels.cuh" 
#include <random>
#include <iostream>

Tensor::Tensor(std::vector<int> shape, bool requires_grad) 
    : shape(shape), requires_grad(requires_grad) {
    
    size = 1;
    for (int s : shape) size *= s;

    // Alocare memorie GPU
    CUDA_CHECK(cudaMalloc(&data, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad, size * sizeof(float)));
    
    // Inițializare grad cu 0
    zero_grad();
}

Tensor::~Tensor() {
    if (data) cudaFree(data);
    if (grad) cudaFree(grad);
}

void Tensor::zero_grad() {
    LaunchFill(grad, 0.0f, size);
}

std::shared_ptr<Tensor> Tensor::zeros(std::vector<int> shape, bool requires_grad) {
    auto t = std::make_shared<Tensor>(shape, requires_grad);
    LaunchFill(t->data, 0.0f, t->size);
    return t;
}

std::shared_ptr<Tensor> Tensor::randn(std::vector<int> shape, bool requires_grad) {
    auto t = std::make_shared<Tensor>(shape, requires_grad);
    
    std::vector<float> host_data(t->size);
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 0.1f);

    for (int i = 0; i < t->size; ++i) host_data[i] = dist(gen);
    
    CUDA_CHECK(cudaMemcpy(t->data, host_data.data(), t->size * sizeof(float), cudaMemcpyHostToDevice));
    return t;
}

void Tensor::to_cpu(float* dest) {
    CUDA_CHECK(cudaMemcpy(dest, data, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// --- AUTOGRAD OPERATIONS ---

std::shared_ptr<Tensor> Tensor::matmul(std::shared_ptr<Tensor> other) {
    int M = this->shape[0];
    int K = this->shape[1];
    int N = other->shape[1]; 

    auto result = std::make_shared<Tensor>(std::vector<int>{M, N}, this->requires_grad || other->requires_grad);

    LaunchMatMul(this->data, other->data, result->data, M, N, K);

    if (result->requires_grad) {
        result->grad_fn = [result, self=shared_from_this(), other, M, N, K]() {
            LaunchMatMulBackward(result->grad, self->data, other->data, 
                                 self->requires_grad ? self->grad : nullptr, 
                                 other->requires_grad ? other->grad : nullptr, 
                                 M, N, K);
            
            if (self->grad_fn) self->grad_fn();
            if (other->grad_fn) other->grad_fn();
        };
    }

    return result;
}

std::shared_ptr<Tensor> Tensor::add(std::shared_ptr<Tensor> other) {
    int rows = this->shape[0];
    int cols = this->shape[1];

    auto result = std::make_shared<Tensor>(this->shape, this->requires_grad || other->requires_grad);

    LaunchAdd(this->data, other->data, result->data, rows, cols);

    if (result->requires_grad) {
        result->grad_fn = [result, self=shared_from_this(), other, rows, cols]() {
            LaunchAddBackward(result->grad, 
                              self->requires_grad ? self->grad : nullptr, 
                              other->requires_grad ? other->grad : nullptr, 
                              rows, cols);
            
            if (self->grad_fn) self->grad_fn();
            if (other->grad_fn) other->grad_fn();
        };
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::relu() {
    auto result = std::make_shared<Tensor>(this->shape, this->requires_grad);
    
    LaunchReLU(this->data, result->data, this->size);

    if (result->requires_grad) {
        result->grad_fn = [result, self=shared_from_this()]() {
            LaunchReLUBackward(self->data, result->grad, self->grad, self->size);
            if (self->grad_fn) self->grad_fn();
        };
    }
    return result;
}

void Tensor::backward() {
    LaunchFill(this->grad, 1.0f, 1);
    
    if (this->grad_fn) {
        this->grad_fn();
    }
}