#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "stdio.h"
#include "time.h"
#include <cstdint>

constexpr int LOOPNUM = 100;

template<typename T>
__device__ __forceinline__ void fastAtomicAdd(T *data, T value) {
    atomicAdd(data, value);
}

template<>
__device__ __forceinline__ void fastAtomicAdd<half>(half *address, half value) {
#if 0
    half *target_addr = reinterpret_cast<half*>(address);
    bool low_byte = (reinterpret_cast<std::uintptr_t>(target_addr)
                     % sizeof(half2) == 0);
    if(low_byte) {
        half2 value2;
        value2.x = value;
        value2.y = __int2half_rz(0);
        atomicAdd(reinterpret_cast<half2*>(target_addr), value2);
    } else {
        half2 value2;
        value2.x = __int2half_rz(0);
        value2.y = value;
        atomicAdd(reinterpret_cast<half2*>(target_addr - 1), value2);
    }
#else
    unsigned int *address_as_ui =
        (unsigned int *)((char*)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    half hsum;
    do {
        assumed = old;
        hsum = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        hsum += value;
        old = (size_t)address & 2 ? (old & 0xffff) |
              (__half2ushort_rn(hsum) << 16) : (old & 0xffff0000) |
              __half2ushort_rn(hsum);
        old = atomicCAS(address_as_ui, assumed, old);
    } while(assumed != old);
    hsum = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
#endif
}

template<typename T>
__global__ void KeAtomic(T *data, int size, T *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        fastAtomicAdd(&out[idx], data[idx]);
    }
}

template<typename T>
float convert2float(T input) {
    return static_cast<float>(input);
}

template<>
float convert2float<half>(half input) {
    return __half2float(input);
}

template<typename TYPE>
float TimeOfKernel(int size, cudaStream_t &context) {
    TYPE *table_h, *table_d;
    cudaMallocHost((void**)&table_h, size * sizeof(TYPE));
    cudaMalloc((void**)&table_d, size * sizeof(TYPE));

    for(int i = 0; i < size; i ++) {
        table_h[i] = 0.00001f * i;
    }
    cudaMemcpyAsync(table_d, table_h, size * sizeof(TYPE), cudaMemcpyHostToDevice, context);

    TYPE *output_h, *output_d;
    cudaMallocHost(&output_h, size * sizeof(TYPE));
    cudaMalloc(&output_d, size * sizeof(TYPE));
    cudaMemset(output_d, 0, size * sizeof(TYPE));

    int threads = 1024;
    int grids = (size + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, context);
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++)
    KeAtomic<TYPE><<<grids, threads, 0, context>>>(
          table_d, size, output_d);
    cudaEventRecord(stop, context);
    cudaEventSynchronize(stop);

    float time_of_kernel;
    cudaEventElapsedTime(&time_of_kernel, start, stop);

#if 0
    cudaMemcpyAsync(output_h, output_d, size * sizeof(TYPE), cudaMemcpyDeviceToHost, context);
    cudaStreamSynchronize(context);
    for(int i = 0; i < 10; i++) {
        printf("%f ", convert2float<TYPE>(output_h[i]));
    }
    printf("\n");
#endif

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(table_d);
    cudaFreeHost(table_h);
    cudaFree(output_d);
    cudaFreeHost(output_h);

    return time_of_kernel;
}

int main() {
    srand((unsigned)time(NULL));
    cudaStream_t context;
    cudaStreamCreate(&context);

    const int beg = 1024;
    const int end = beg << 16;
    for(int i = beg; i <= end; i <<= 1) {
        float t_fp13 = TimeOfKernel<float>(i, context);
        float t_fp16 = TimeOfKernel<half>(i, context);
        printf("atomic %d time fp32 %f ms vs fp16 %f ms\n", i, t_fp13, t_fp16);
    }

    cudaStreamDestroy(context);
    return 0;
}