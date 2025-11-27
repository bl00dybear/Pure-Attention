#include "optim/Optimizer.h"
#include "backend/Kernels.cuh"

Adam::Adam(std::vector<std::shared_ptr<Tensor>> params, float lr, float beta1, float beta2)
    : parameters(params), lr(lr), beta1(beta1), beta2(beta2), epsilon(1e-8), t(0) {
    
    // Inițializare m și v pentru fiecare parametru
    for (auto p : parameters) {
        float* m;
        float* v;
        CUDA_CHECK(cudaMalloc(&m, p->size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&v, p->size * sizeof(float)));
        LaunchFill(m, 0.0f, p->size);
        LaunchFill(v, 0.0f, p->size);
        
        state[p.get()] = {m, v};
    }
}

Adam::~Adam() {
    for (auto& kv : state) {
        cudaFree(kv.second.m_device);
        cudaFree(kv.second.v_device);
    }
}

void Adam::zero_grad() {
    for (auto p : parameters) {
        p->zero_grad();
    }
}

void Adam::step() {
    t++;
    for (auto p : parameters) {
        AdamState& s = state[p.get()];
        LaunchAdamStep(p->data, p->grad, s.m_device, s.v_device, 
                       p->size, t, lr, beta1, beta2, epsilon);
    }
}