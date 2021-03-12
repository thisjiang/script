#include <cuda_runtime.h>
#include <cuda_fp16.h>
// Primary header is compatible with pre-C++11, collective algorithm headers require C++11
#include <cooperative_groups.h>
#include "stdio.h"

#define WARP_SIZE 32

__global__ void KeSum(float *data, int size, float* res) {
    if(threadIdx.x == 0) {
        float tmp = 0.0f;
        for(int i = 0; i < size; i++) {
            tmp += data[i];
        }
        (*res) = tmp;
    }
}

//Simple version
__global__ void KeReduceSumSimple(float* data, int size, 
                            float* block_res, int block_size) {
    const int idx = blockIdx.x * block_size + threadIdx.x;
    const int block_end = min((blockIdx.x + 1) * block_size, size);
    
    if(idx < block_end) {
        extern __shared__ float thread_res[];

        float res = 0.0f;
        for(int i = idx; i < block_end; i += blockDim.x) {
            res += data[i];
        }
        thread_res[threadIdx.x] = res;
        __syncthreads();
        
        if(threadIdx.x < 32) {
            res = 0.0f;
            for(int i = threadIdx.x; i < blockDim.x; i += WARP_SIZE) {
                res += thread_res[i];
            }            
            thread_res[threadIdx.x] = res;  
        }        
        __syncwarp();

        if(threadIdx.x == 0) {
            res = 0.0f;
            for(int i = 0; i < WARP_SIZE && i < blockDim.x; i ++) {
                res += thread_res[i];
            }
            block_res[blockIdx.x] = res;
        }        
    }
}
//再起另一个kernel or 拷贝到host端计算最终结果

//Shuffle version
__device__ __forceinline__ float ShuffleReduceSum(float value) {
    for(int i = 16; i > 0; i /= 2) {
        value += __shfl_xor_sync(0xffffffff, value, i, WARP_SIZE);
    }
    return value;
}

__global__ void KeReduceSumShuffle(float* data, int size, 
                            float* block_res, int block_size) {
    const int idx = blockIdx.x * block_size + threadIdx.x;
    const int block_end = min((blockIdx.x + 1) * block_size, size);
    const int warp_num = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    if(idx < block_end) {
        extern __shared__ float warp_res[];

        float res = 0.0f;
        for(int i = idx; i < block_end; i += blockDim.x) {
            res += data[i];
        }
        warp_res[threadIdx.x / WARP_SIZE] = ShuffleReduceSum(res);

        __syncthreads();

        if(threadIdx.x < 32) { //first warp
            res = 0.0f;
            for(int i = threadIdx.x; i < warp_num; i += WARP_SIZE) {
                res += warp_res[i];
            }
            block_res[blockIdx.x] = ShuffleReduceSum(res);
        }
    }
}
//再起另一个kernel or 拷贝到host端计算最终结果


//Shuffle version with cooperative_groups
__global__ void KeReduceSumCg(float* data, int size, 
                            float* block_res, int block_size,
                            float* final_res) {
    const int idx = blockIdx.x * block_size + threadIdx.x;
    const int block_end = min((blockIdx.x + 1) * block_size, size);
    const int warp_num = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    if(idx < block_end) {
        extern __shared__ float warp_res[];

        float res = 0.0f;
        for(int i = idx; i < block_end; i += blockDim.x) {
            res += data[i];
        }
        warp_res[threadIdx.x / WARP_SIZE] = ShuffleReduceSum(res);

        __syncthreads();

        if(threadIdx.x < 32) { //first warp
            res = 0.0f;
            for(int i = threadIdx.x; i < warp_num; i += WARP_SIZE) {
                res += warp_res[i];
            }
            block_res[blockIdx.x] = ShuffleReduceSum(res);
        }

        cooperative_groups::this_grid().sync();

        if(threadIdx.x < 32) { //first warp
            res = 0.0f;
            for(int i = threadIdx.x; i < gridDim.x; i += WARP_SIZE) {
                res += block_res[i];
            }
            (*final_res) = ShuffleReduceSum(res);
        }
    }
}

int main() {
    constexpr int T_SIZE = sizeof(float);
    size_t size = 2048*1024, mem_size = size * T_SIZE;
    int block = 512, block_size = block * 16;
    int grid = size / block_size, bmem_size = grid * T_SIZE;
    int warp_num = (block + WARP_SIZE - 1) / WARP_SIZE;

    float real_res = 0.0f;

    float *data_h, *data_d;
    cudaMalloc(&data_d, mem_size);
    cudaMallocHost(&data_h, mem_size);

    for(int i = 0; i < size; i ++) {
        real_res += i;
        data_h[i] = i;
    }
    cudaMemcpy(data_d, data_h, mem_size, cudaMemcpyHostToDevice);

    float *block_d, *block_h;
    cudaMalloc(&block_d, bmem_size);
    cudaMemset(block_d, 0, bmem_size);
    cudaMallocHost(&block_h, bmem_size);

    float *res_d, *res_h;
    cudaMalloc(&res_d, T_SIZE);
    cudaMallocHost(&res_h, T_SIZE);
    *res_h = 0.0f;

    //KeReduceSumSimple<<<grid, block, block * T_SIZE>>>(data_d, size, block_d, block_size);
    //KeReduceSumSimple<<<1, grid, grid * T_SIZE>>>(block_d, grid, res_d, grid);
    
    KeReduceSumShuffle<<<grid, block, warp_num * T_SIZE>>>(data_d, size, block_d, block_size);
    KeReduceSumShuffle<<<1, grid, (grid + 31) / 32 * T_SIZE>>>(block_d, grid, res_d, grid);

    //KeReduceSumCg<<<grid, block, warp_num * T_SIZE>>>(data_d, size, block_d, block_size, res_d);
    //void *args[] = {(void*)&data_d, (void*)&size, (void*)&block_d, (void*)&block_size, (void*)&res_d};
    //cudaLaunchCooperativeKernel((void*)KeReduceSumCg, grid, block, args, warp_num * T_SIZE);

    cudaDeviceSynchronize();

    cudaMemcpy(res_h, res_d, T_SIZE, cudaMemcpyDeviceToHost);
    printf("%f %f\n", real_res, *res_h);

    return 0;
}