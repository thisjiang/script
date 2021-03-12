#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "stdio.h"

//simple version
__global__ void KeMatrixTransposeSimple(float* A, float* B, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < n) {
        B[col * m + row] = A[row * n + col];
    }
}
//matrix B not coalesced

//shared version
__global__ void KeMatrixTransposeShared(float* A, float* B, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < m && col < n) {
        extern __shared__ float s_data[];

        s_data[row * n + col] = A[row * n + col];
        __syncthreads();
        B[row * n + col] = s_data[col * m + row];
    }
}

//shared version with padding 1
__global__ void KeMatrixTransposePad1(float* A, float* B, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < n) {
        extern __shared__ float s_pad[];

        s_pad[row * (n + 1) + (col + row) % n] = A[row * n + col];
        __syncthreads();
        B[row * n + col] = s_pad[col * (n + 1) + (row + col) % n];
    }
}

int main() {
    size_t size = 1024 * sizeof(float);

    float *data_h;
    cudaMallocHost(&data_h, size);
    for(int i = 0; i < 1024; i ++) data_h[i] = i;

    float *data_d;
    cudaMalloc(&data_d, size);

    cudaMemcpy(data_d, data_h, size, cudaMemcpyHostToDevice);

    float *res_d;
    cudaMalloc(&res_d, size);

    dim3 grid(1, 1), block(32, 32);

    KeMatrixTransposeSimple<<<grid, block>>>(data_d, res_d, 32, 32);
    KeMatrixTransposeShared<<<grid, block, size>>>(data_d, res_d, 32, 32);
    KeMatrixTransposePad1<<<grid, block, size + 32 * sizeof(float)>>>(data_d, res_d, 32, 32);

    cudaDeviceSynchronize();

    float *res_h;
    cudaMallocHost(&res_h, size);
    cudaMemcpy(res_h, res_d, size, cudaMemcpyDeviceToHost);

    // for(int i = 0; i < 32; i ++) {
    //     for(int j = 0; j < 32; j ++) {
    //         printf("%f ", res_h[i * 32 +j]);
    //     }
    //     printf("\n");
    // }

    return 0;
}