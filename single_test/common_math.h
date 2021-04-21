#ifndef SCRIPT_COMMON_MATH_H
#define SCRIPT_COMMON_MATH_H

#include "math.h"
#include "limits.h"
#include "float.h"
#include "ctype.h"

#include "cuda_runtime.h"
#include "cuda_fp16.h"

#include <vector>
#include <array>
#include <type_traits>
#include <limits>

/***********************************************************/

#define HOSTDEVICE __forceinline__ __device__ __host__

/***********************************************************/

typedef half float16;

namespace paddle {
namespace platform {
  typedef half float16;
}
}

/***********************************************************/

template <typename T>
struct GetAccType {
  using type = T;
};
template <>
struct GetAccType<paddle::platform::float16> {
  using type = float;
};

template <typename T, int N>
struct GetVecType;
template <typename T>
struct GetVecType<T, 1> {
  using type = T;
};
template <>
struct GetVecType<paddle::platform::float16, 2> {
  using type = half2;
};
template <>
struct GetVecType<paddle::platform::float16, 4> {
  using type = float2;
};
template <>
struct GetVecType<float, 2> {
  using type = float2;
};
template <>
struct GetVecType<float, 4> {
  using type = float4;
};
template <>
struct GetVecType<double, 2> {
  using type = double2;
};
template <>
struct GetVecType<double, 4> {
  using type = double4;
};

/***********************************************************/

template <typename T, int Size, T DefaultValue>
struct __align__(sizeof(T)) Array {
    HOSTDEVICE const T& operator[](int index) const {
        return data[index];
    }
    HOSTDEVICE T& operator[](int index) {
        return data[index];
    }

    HOSTDEVICE Array() {
#pragma unroll
        for(int i = 0; i < Size; i ++) data[i] = DefaultValue;
    }

    HOSTDEVICE Array(const std::initializer_list<T> &arr) {
        int i = 0;
        for(T value : arr) data[i ++] = value;
#pragma unroll
        for(; i < Size; i ++) data[i] = DefaultValue;
    }

    HOSTDEVICE Array(const Array &arr) {
#pragma unroll
        for(int i = 0; i < Size; i ++) data[i] = arr.data[i];
    }

    T data[Size];
};

struct Dim3 : Array<size_t, 3, 1> {
    HOSTDEVICE Dim3() : Array<size_t, 3, 1>() {}
    HOSTDEVICE Dim3(size_t x) : Array<size_t, 3, 1>({x}) {}
    HOSTDEVICE Dim3(size_t x, size_t y) : Array<size_t, 3, 1>({x, y}) {}
    HOSTDEVICE Dim3(size_t x, size_t y, size_t z)
        : Array<size_t, 3, 1>({x, y, z}) {}
};

typedef std::vector<int> DDim;

/***********************************************************/

template<typename T>
__forceinline__ __device__ T Abs(const T val) {
  return abs(val);
}
template<>
__forceinline__ __device__ half Abs(const half val) {
  return __hlt(val, (half)0) ? __hneg(val) : val;
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
static HOSTDEVICE size_t GetSize(const dim3 &dims) {
    return dims.x * dims.y * dims.z;
}

static HOSTDEVICE size_t GetSize(const Dim3 &dims) {
    return dims[0] * dims[1] * dims[2];
}

template<typename T>
static inline size_t GetSize(const std::vector<T> &dims) {
    size_t res = 1;
    for(auto d : dims) res *= d;
    return res;
}

template<typename T, size_t D>
static inline size_t GetSize(const std::array<T, D> &dims) {
    size_t res = 1;
    for(auto d : dims) res *= d;
    return res;
}

template<typename T>
static HOSTDEVICE size_t GetSize(const T *dims, int n) {
    size_t res = 1;
    for(int i = 0; i < n; i ++) res *= dims[i];
    return res;
}

/***********************************************************/
static HOSTDEVICE size_t GetSum(const dim3 &dims) {
    return dims.x + dims.y + dims.z;
}

static HOSTDEVICE size_t GetSum(const Dim3 &dims) {
    return dims[0] + dims[1] + dims[2];
}

template<typename T>
static inline size_t GetSum(const std::vector<T> &dims) {
    size_t res = 0;
    for(auto d : dims) res += d;
    return res;
}

template<typename T, size_t D>
static inline size_t GetSum(const std::array<T, D> &dims) {
    size_t res = 0;
    for(auto d : dims) res += d;
    return res;
}

template<typename T>
static HOSTDEVICE size_t GetSum(const T *dims, int n) {
    size_t res = 0;
    for(int i = 0; i < n; i ++) res += dims[i];
    return res;
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