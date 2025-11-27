#include "backend/Kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define BLOCK_SIZE 16

// ==========================================
// KERNELS (__global__) - Rulează pe GPU
// ==========================================

__global__ void MatMulKernel(const float* A, const float* B, float* C, 
                             int M, int N, int K, 
                             bool transA, bool transB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            int idxA = transA ? (k * M + row) : (row * K + k);
            int idxB = transB ? (col * K + k) : (k * N + col);
            sum += A[idxA] * B[idxB];
        }
        C[row * N + col] = sum;
    }
}

__global__ void AddBiasKernel(const float* A, const float* B, float* C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = rows * cols;
    if (idx < size) {
        int col = idx % cols; // Broadcasting
        C[idx] = A[idx] + B[col];
    }
}

__global__ void AddBiasBackwardKernel(const float* grad_output, float* grad_bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        atomicAdd(&grad_bias[col], grad_output[idx]);
    }
}

__global__ void ReLUKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void ReLUBackwardKernel(const float* input, const float* grad_output, float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

__global__ void MSELossKernel(const float* preds, const float* targets, float* loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = preds[idx] - targets[idx];
        float val = (diff * diff) / (float)size;
        atomicAdd(loss, val);
    }
}

__global__ void MSELossBackwardKernel(const float* preds, const float* targets, float* grad_preds, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_preds[idx] = 2.0f * (preds[idx] - targets[idx]) / (float)size;
    }
}

__global__ void AdamStepKernel(float* param, float* grads, float* m, float* v, 
                               int size, int step, float lr, float beta1, float beta2, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grads[idx];
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float m_hat = m[idx] / (1.0f - powf(beta1, step));
        float v_hat = v[idx] / (1.0f - powf(beta2, step));
        
        param[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

__global__ void FillKernel(float* data, float value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = value;
}

// ==========================================
// HOST WRAPPERS (extern "C") - Apelabile din C++
// ==========================================

extern "C" {
    void LaunchMatMul(const float* A, const float* B, float* C, int M, int N, int K) {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        MatMulKernel<<<grid, block>>>(A, B, C, M, N, K, false, false);
    }

    void LaunchMatMulBackward(const float* grad_C, const float* A, const float* B, 
                              float* grad_A, float* grad_B, int M, int N, int K) {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        if(grad_A) {
             dim3 gridA((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);
             MatMulKernel<<<gridA, block>>>(grad_C, B, grad_A, M, K, N, false, true); 
        }
        if(grad_B) {
            dim3 gridB((N + block.x - 1) / block.x, (K + block.y - 1) / block.y);
            MatMulKernel<<<gridB, block>>>(A, grad_C, grad_B, K, N, M, true, false);
        }
    }

    void LaunchAdd(const float* A, const float* B, float* C, int rows, int cols) {
        int size = rows * cols;
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        AddBiasKernel<<<blocks, threads>>>(A, B, C, rows, cols);
    }

    void LaunchAddBackward(const float* grad_C, float* grad_A, float* grad_B, int rows, int cols) {
        int threads = 256;
        int blocks = (rows * cols + threads - 1) / threads;
        
        if (grad_A) cudaMemcpy(grad_A, grad_C, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice);
        
        if (grad_B) {
            int fill_blocks = (cols + threads - 1) / threads;
            FillKernel<<<fill_blocks, threads>>>(grad_B, 0.0f, cols); 
            AddBiasBackwardKernel<<<blocks, threads>>>(grad_C, grad_B, rows, cols);
        }
    }

    void LaunchReLU(const float* input, float* output, int size) {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        ReLUKernel<<<blocks, threads>>>(input, output, size);
    }

    void LaunchReLUBackward(const float* input, const float* grad_output, float* grad_input, int size) {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        ReLUBackwardKernel<<<blocks, threads>>>(input, grad_output, grad_input, size);
    }

    void LaunchMSELoss(const float* preds, const float* targets, float* loss_out, int size) {
        LaunchFill(loss_out, 0.0f, 1);
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        MSELossKernel<<<blocks, threads>>>(preds, targets, loss_out, size);
    }

    void LaunchMSELossBackward(const float* preds, const float* targets, float* grad_preds, int size) {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        MSELossBackwardKernel<<<blocks, threads>>>(preds, targets, grad_preds, size);
    }

    void LaunchAdamStep(float* param, float* grads, float* m, float* v, 
                        int size, int step, float lr, float beta1, float beta2, float epsilon) {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        AdamStepKernel<<<blocks, threads>>>(param, grads, m, v, size, step, lr, beta1, beta2, epsilon);
    }

    void LaunchFill(float* data, float value, int size) {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        FillKernel<<<blocks, threads>>>(data, value, size);
    }
}