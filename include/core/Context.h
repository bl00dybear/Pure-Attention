#pragma once

// libs
#include <cuda_runtime.h>

struct CudaContext {
    static cudaStream_t current_stream;
    
    static cudaStream_t getStream() {
        return current_stream;
    }
    
    static void setStream(cudaStream_t s) {
        current_stream = s;
    }
};