#ifndef SCRIPT_COMMON_MATH_H
#define SCRIPT_COMMON_MATH_H

#include "math.h"

#include "cuda_runtime.h"
#include "cuda_fp16.h"

/***********************************************************/

#define HOSTDEVICE __forceinline__ __device__ __host__

/***********************************************************/

typedef half float16;

/***********************************************************/

template<typename T>
__forceinline__ __device__ T Abs(const T val) {
  return abs(val);
}
template<>
__forceinline__ __device__ half Abs(const half val) {
  return __hlt(val, 0) ? __hneg(val) : val;
}

/***********************************************************/

template<typename T>
__forceinline__ __device__ T Exp(const T val) {
  return exp(val);
}
template<>
__forceinline__ __device__ float Exp<float>(const float val) {
  return __expf(val);
}
template<>
__forceinline__ __device__ half Exp<half>(const half val) {
  return hexp(val);
}

/***********************************************************/

template<typename INTYPE, typename OUTTYPE>
HOSTDEVICE OUTTYPE type2type(INTYPE val) {
    return static_cast<OUTTYPE>(val);
}
template<>
HOSTDEVICE float type2type<float16, float>(float16 val) {
    return __half2float(val);
}
template<>
HOSTDEVICE float16 type2type<float, float16>(float val) {
    return __float2half(val);
}

template<typename T1, typename T2>
inline void ConvertHost(T1 *des, T2 *src, size_t num) {
    for(int i = 0; i < num; i ++) {
        des[i] = type2type<T2, T1>(src[i]);
    }
}

template<typename T1, typename T2>
__global__ void KeConvert(T1 *des, T2 *src, size_t num) {
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = idx; i < num; i += blockDim.x * gridDim.x) {
        des[i] = type2type<T2, T1>(src[i]);
    }
}

template<typename T1, typename T2>
inline void ConvertDevice(T1 *des, T2 *src, size_t num, 
                          cudaStream_t &stream) {
    int block = std::min(num, static_cast<size_t>(256));
    int gird = (num + block - 1) / block;
    KeConvert<<<gird, block, 0, stream>>>(des, src, num);
}

/***********************************************************/




#endif // SCRIPT_COMMON_MATH_H