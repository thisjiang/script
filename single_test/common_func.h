#include "cuda_runtime.h"
#include "cuda_fp16.h"

#include "stdio.h"

template<typename IN, typename OUT>
__forceinline__ __device__ __host__ OUT type2type(IN val) {
    return static_cast<OUT>(val);
}
template<>
__forceinline__ __device__ __host__ float type2type<half, float>(half val) {
    return __half2float(val);
}
template<>
__forceinline__ __device__ __host__ half type2type<float, half>(float val) {
    return __float2half(val);
}

template<typename T1, typename T2>
__global__ void KeConvertDevice(T1 *des, T2 *src, size_t num) {
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = idx; i < num; i += blockDim.x * gridDim.x) {
        des[i] = type2type<T2, T1>(src[i]);
    }
}

template<typename T1, typename T2>
inline void KeConvertHost(T1 *des, T2 *src, size_t num) {
    for(int i = 0; i < num; i ++) {
        des[i] = type2type<T2, T1>(src[i]);
    }
}

template<typename T1, typename T2>
double MaxError(T1 *data_1, T2 *data_2, int64_t size) {
    double maxerr = 0.0, err = 0.0;
    for(int i = 0; i < size; i ++) {
        err = fabs(type2type<T1, float>(data_1[i]) - type2type<T2, float>(data_2[i]));
        if(err > maxerr) maxerr = err;
    }
    return maxerr;
}

template<typename T> void print_ele(T val) {}

#define PRINT_INT(T)                        \
    template<> void print_ele<T>(T val) {   \
        printf("%d", type2type<T, int>(val)); \
    }

PRINT_INT(int)
PRINT_INT(size_t)
PRINT_INT(int8_t)
PRINT_INT(bool)

#undef PRINT_INT

#define PRINT_FLOAT(T)                        \
    template<> void print_ele<T>(T val) {     \
        printf("%f", type2type<T, float>(val)); \
    }

PRINT_FLOAT(float)
PRINT_FLOAT(half)

#undef PRINT_FLOAT

template<> void print_ele<double>(double val) {printf("%f", val);}
template<> void print_ele<int64_t>(int64_t val) {printf("%ld", val);}