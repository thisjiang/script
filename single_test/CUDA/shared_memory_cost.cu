#include "../common.h"

#include "cuda_runtime.h"
#include "cuda_fp16.h"

#include "time.h"


template<typename T, int SIZE>
__global__ void KeStaticShared() {
    __shared__ T s_data[SIZE];
}

template<typename T>
__global__ void KeDynamicShared() {
    extern __shared__ T s_data[];
}

int main() {
    cudaStream_t context;
    cudaStreamCreate(&context);
 
    TimeOfKernel* sat = TimeOfKernel::get();
    sat->start(context);
    KeDynamicShared<float><<<1, 1, sizeof(float), context>>>();
    printf("Dynamic %f ms\n", sat->stop(context));

    sat->start(context);
    KeStaticShared<float, 1><<<1, 1, 0, context>>>();
    printf("Static %f ms\n", sat->stop(context));

    cudaStreamDestroy(context);
    return 0;
}