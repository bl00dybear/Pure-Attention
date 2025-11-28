#include <cuda_runtime.h>
#include <core/Tensor.h>

#define TILE_WIDTH 16

__global__ void matmul_kernel_tiled(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    // Global partitionare (tile-uri):
    // A:                 B:
    // +---+---+---+     +---+---+---+
    // |A00|A01|A02| ... |B00|B01|B02|
    // +---+---+---+     +---+---+---+
    // |A10|A11|A12|     |B10|B11|B12|
    // +---+---+---+     +---+---+---+

    // C_ij = Σ_m A(i,m) * B(m,j) => pentru un tile C(i,j) se parcurg m=0..M-1 și se incarca A(i,m), B(m,j)

    // T=4, mapping thread-uri -> shared:
    // Global coords:
    // row = i*T + ty
    // col = j*T + tx

    // Shared tiles:
    // s_A (ty x tx)        s_B (ty x tx)
    // [ (0,0) (0,1) (0,2) (0,3) ]   [ (0,0) (0,1) (0,2) (0,3) ]
    // [ (1,0) (1,1) (1,2) (1,3) ]   [ (1,0) (1,1) (1,2) (1,3) ]
    // [ (2,0) (2,1) (2,2) (2,3) ]   [ (2,0) (2,1) (2,2) (2,3) ]
    // [ (3,0) (3,1) (3,2) (3,3) ]   [ (3,0) (3,1) (3,2) (3,3) ]

    // Fiecare thread (ty,tx):
    // s_A[ty][tx] = A[row, m*T + tx]
    // s_B[ty][tx] = B[m*T + ty, col]

    // Compute:
    // for k in 0..T-1:
    //      val += s_A[ty][k] * s_B[k][tx]
    
    // bx si by reprezinta indexii blocului curent, iar blocurile le am mapat
    // la tile urile in care spargem matricile mari
    // tx si ty sunt coordonatele primite de thread ul curent, coordonate locale 
    // raportate la blocul curent
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // folosind adresele blocului si thread ului curent recalculam adresa globala 
    // pe care vom scrie rezultatul dupa inmultire
    int global_row = blockIdx.y * TILE_WIDTH + ty;
    int global_col = blockIdx.x * TILE_WIDTH + tx;

    // 2048/16=128 tile uri
    // 2050/16=128 (ne trebuie 129) asa ca facem
    // (2050+16-1)/16=2065/16=129
    // (2048+16-1)/16=2063/16=128
    int tile_num_reduction=(N+TILE_WIDTH-1)/TILE_WIDTH;
    float val = 0.0f;

    for(int m = 0; m < tile_num_reduction; m += 1) {
        int global_read_row_A = global_row;
        int global_read_col_A = m * TILE_WIDTH + tx;

        int global_read_row_B = m * TILE_WIDTH + ty;
        int global_read_col_B = global_col;
        
        if(global_read_col_A < N && global_read_row_A < M) {
            s_A[ty][tx] = A[global_read_row_A * N + global_read_col_A];
        }
        else {
            s_A[ty][tx] = 0.0f;
        }

        if(global_read_col_B < K && global_read_row_B < N) {
            s_B[ty][tx] = B[global_read_row_B * K + global_read_col_B];
        }
        else {
            s_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            val += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    } 

    if (global_row < M && global_col < K) {
        C[global_row * K + global_col] = val;
    }
}

__global__ void addMatrixVector(const float *A, const float *X, float *B, const int M, const int N) {
    int global_col = blockIdx.x * blockDim.x + threadIdx.x;
    int global_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (global_row < M && global_col < N) {
        int index = global_row * N + global_col;
        B[index] = A[index] + X[global_col];
    }
}
























void launch_matmul_tiled(core::Tensor& A, core::Tensor& B, core::Tensor& C,cudaStream_t stream) {
    int M = A.get_shape()[0];
    int N = A.get_shape()[1];
    int K = B.get_shape()[1];

    // definim un bloc de thread uri cu 16x16 threads per block (TILE_WIDTH=16)
    dim3 block(TILE_WIDTH, TILE_WIDTH); 
    // pt N=2048, grid ul va fi 2048/16=128, deci avem un grid de (128,128), adica 128x128 
    // blocuri => nr total de thread de executie = 128x128 x 16x16
    dim3 grid((K+block.x-1)/block.x,(M+block.y-1)/block.y);

    // software: thread ul este cea mai mica unitate de calcul 
    // hardware: cea mai mica unitate de calcul este warp-ul, care contine o gasca de thread uri
    // (de obicei 32) si practic warp urile se executa pe SM.
    // fiecare warp are registrii lui pe cip si deci context switch se face instantaneu (nu ca pe cpu unde 
    // scrii registrii in ram).
    // un block sta pe sm pana la finalizare, dar pe acelasi sm se pot pune mai multe blockuri (fiecare SM are 
    // un warp pool)

    // memoria shared este share uita de toate warp urile de pe un SM, dar sa fie warp urile 
    // aceluias block, clar mult mai rapida decat global (vram)
    // dynamic shared memory: 0 indica cate bytes de memorie shared extra alocam per bloc 
    // la niveld e memorie astea sunt singurele 2 pe care le controlam explicit
    // implicit ar fi 2 comportamente de luat in considerare:
    // - aliniere si coalescing (modul in care indexezi trebuie sa faca ca thread urile unui wrap
    //      sa citeasca adrese consecutive)
    // - reuse local maxim prin tile uri in shared (fiecare tile este adus o singura data din memoria globala
    //      si thread urile blocului folosesc repetat datele aduse in shared)
    matmul_kernel_tiled<<<grid, block, 0, stream>>>(A.get_data_ptr(), B.get_data_ptr(), C.get_data_ptr(), M, N, K);
}

void launch_addMatrixVector_tiled