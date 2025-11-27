#pragma once

extern "C" {
    // Basic Maths (Updated for broadcasting)
    void LaunchAdd(const float* A, const float* B, float* C, int rows, int cols);
    void LaunchAddBackward(const float* grad_C, float* grad_A, float* grad_B, int rows, int cols);
    
    // Matrix Multiplication
    void LaunchMatMul(const float* A, const float* B, float* C, int M, int N, int K);
    void LaunchMatMulBackward(const float* grad_C, const float* A, const float* B, 
                              float* grad_A, float* grad_B, int M, int N, int K);

    // Activations
    void LaunchReLU(const float* input, float* output, int size);
    void LaunchReLUBackward(const float* input, const float* grad_output, float* grad_input, int size);

    // Loss
    void LaunchMSELoss(const float* preds, const float* targets, float* loss_out, int size);
    void LaunchMSELossBackward(const float* preds, const float* targets, float* grad_preds, int size);

    // Optimizer
    void LaunchAdamStep(float* param, float* grads, float* m, float* v, 
                        int size, int step, float lr, float beta1, float beta2, float epsilon);
    
    // Utils
    void LaunchFill(float* data, float value, int size);
}