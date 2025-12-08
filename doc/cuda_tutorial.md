## Cuda first steps:

```C
#include <iostream>
#include <cuda_runtime.h>

__global__ void addMatrixVector(const float *A, const float *X, float*B, const int rows, const int cols) {
    // Calculate the global column and row indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (row < rows && col < cols) {
        int index = row * cols + col;
        B[index] = A[index] + X[col];
    }
}

int main() {
    // init N
    int rows = 4;
    int cols = 5;
    size_t size_matrix = rows * cols * sizeof(float);
    size_t size_vector = cols * sizeof(float);

    // HOST
    // declare and init vect
    float *A = new float[rows * cols];
    float *X = new float[cols];
    float *B = new float[rows * cols];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            A[i * cols + j] = i + j;
        }
    }

    for (int i = 0; i < cols; i++) {
        X[i] = 0.0f;
    }

    // DEVICE
    // memory alloc 
    // ! cudaMalloc
    float *d_A, *d_X, *d_B;

    cudaMalloc((void**)&d_A, size_matrix);
    cudaMalloc((void**)&d_X, size_vector);
    cudaMalloc((void**)&d_B, size_matrix);

    
    // copy from host -> device
    // ! cudaMemcpy
    cudaMemcpy(d_A, A, size_matrix, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, X, size_vector, cudaMemcpyHostToDevice);

    // calculate threadsPerBlock and BlocksPerGrid
    // ! (N + thrds - 1) / thrds
    dim3 block(16, 16);
    dim3 grid(256, 256);

    // call the kernell <<<blocks, threads>>>()
    addMatrixVector<<<grid, block>>>(d_A, d_X, d_B, rows, cols);

    // wait for device
    // ! cudaDeviceSynchronize()
    cudaDeviceSynchronize();

    // copy from device -> host
    cudaMemcpy(B, d_B, size_matrix, cudaMemcpyDeviceToHost);

    // print
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << B[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    // free Device then Host
    cudaFree(d_A); cudaFree(d_X); cudaFree(d_B);
    delete A; delete X; delete B;
}
```