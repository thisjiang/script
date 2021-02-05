#include "stdio.h"
#include "time.h"

#include "cub/cub.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <iostream>
#include <vector>
#include <random>

/***********************************************************/
#define SUCCESS 0
#define CUDA_FAILED -1
#define CHECK_FAILED 1


/***********************************************************/

constexpr int WARP_SIZE = 32;

/***********************************************************/

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
                          cudaStream_t &stream = 0) {
    int block = std::min(num, static_cast<size_t>(256));
    int gird = (num + block - 1) / block;
    KeConvert<<<gird, block, 0, stream>>>(des, src, num);
}

/***********************************************************/
template <typename T>
__forceinline__ __device__ T KeWarpReduceSum(T val, unsigned lane_mask) {
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(lane_mask, val, mask, warpSize);
  return val;
}

template <typename T>
__forceinline__ __device__ T KeWarpReduceMax(T val, unsigned lane_mask) {
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(lane_mask, val, mask, warpSize));
  return val;
}

template <typename T>
__forceinline__ __device__ T KeWarpReduceMin(T val, unsigned lane_mask) {
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
    val = min(val, __shfl_xor_sync(lane_mask, val, mask, warpSize));
  return val;
}

/***********************************************************/
template<typename T>
T MaxErrorHost(const T *a, const T *b, const size_t n) {
    T maxerr = 0.0, err = 0.0;
    for(int i = 0; i < n; i ++) {
        err = a[i] - b[i];
        if(err < 0) err = -err;
        if(err > maxerr) maxerr = err;
    }
    return maxerr;
}

template<typename T, int BLOCKDIM = 256>
__global__ void KeMaxError(const T *a, const T *b, const size_t n, 
                           T *block_max) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  typedef cub::BlockReduce<T, BLOCKDIM> BlockReduce;
  __shared__ typename BlockReduce::TempStorage cub_tmp;

  T max_err(0);
  for(int i = tid; i < n; i += blockDim.x * gridDim.x) {
      T err = a[i] - b[i];
      if(err < 0) err = - err;
      err = BlockReduce(cub_tmp).Reduce(err, cub::Max());

      if(tid == 0) if(err > max_err) max_err = err;
  }
  if(tid == 0) block_max[blockIdx.x] = max_err;
}

template<typename T>
T MaxErrorDevice(const T *a, const T *b, const size_t n,
                cudaStream_t &stream) {
    constexpr int threads = 512;
    int grids = (n + threads - 1) / threads;

    T *block_max;
    cudaMalloc(&block_max, (grids + 1) * sizeof(T));

    void *tmp_mem = nullptr;
    size_t tmp_size = 0;
    cub::DeviceReduce::Max(tmp_mem, tmp_size, block_max, 
                            block_max + grids, grids, stream);
    cudaMalloc(&tmp_mem, tmp_size);

    KeMaxError<T, threads><<<threads, grids, 0, stream>>>
                                (a, b, n, block_max);
    cub::DeviceReduce::Max(tmp_mem, tmp_size, block_max, 
                           block_max + grids, grids, stream);

    T max_err(0);
    cudaMemcpyAsync(&max_err, block_max + grids, sizeof(T), 
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(block_max); // implict sync
    cudaFree(tmp_mem);

    return max_err;
}

/***********************************************************/
template<typename T>
bool CheckSameHost(const T *a, const T *b, size_t n) {
    return memcmp(reinterpret_cast<const void*>(a),
                  reinterpret_cast<const void*>(b), n) == 0;
}

template<typename T>
__global__ void KeCheckSameKernel(const T *a, const T *b, size_t n, int *err_num) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = id; i < n; i += blockDim.x * gridDim.x) {
        if(a[i] != b[i]) {
            atomicAdd(err_num, 1);
        }
    }
}

template<typename T>
bool CheckSameDevice(const T *a, const T *b, size_t n,
                   cudaStream_t &stream) {
    int threads = std::min(n, static_cast<size_t>(256));
    int grids = (n + threads - 1) / threads;

    int *err_d, err_h = 0;
    cudaMalloc(&err_d, sizeof(int));
    cudaMemsetAsync(err_d, 0, sizeof(int), stream);
    KeCheckSameKernel<T><<<threads, grids, 0, stream>>>(a, b, n, err_d);
    cudaMemcpyAsync(&err_h, err_d, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(err_d); // implict sync

    return err_h == 0;
}


/***********************************************************/
template<typename T> void print(T val) {}
template<typename T> void fprint(T val) {}


#define PRINT_INT(T)                        \
  template<> void print<T>(T val) {printf("%d", type2type<T, int>(val));} \
  template<> void fprint<T>(T val) {fprintf(stderr, "%d", type2type<T, int>(val));}

PRINT_INT(int)
PRINT_INT(int8_t)
PRINT_INT(bool)

#undef PRINT_INT

#define PRINT_UINT(T)                        \
  template<> void print<T>(T val) {printf("%u", type2type<T, int>(val));} \
  template<> void fprint<T>(T val) {fprintf(stderr, "%u", type2type<T, int>(val));}

PRINT_UINT(unsigned int)
PRINT_UINT(size_t)
#undef PRINT_UINT

#define PRINT_LONG(T)                        \
  template<> void print<T>(T val) {printf("%lld", type2type<T, int64_t>(val));} \
  template<> void fprint<T>(T val) {fprintf(stderr, "%lld", type2type<T, int64_t>(val));}

PRINT_LONG(int64_t)
PRINT_LONG(long long)
#undef PRINT_LONG

#define PRINT_FLOAT(T)                        \
    template<> void print<T>(T val) {printf("%f", type2type<T, float>(val));} \
    template<> void fprint<T>(T val) {fprintf(stderr, "%f", type2type<T, float>(val));}

PRINT_FLOAT(float)
PRINT_FLOAT(half)
PRINT_FLOAT(double)

#undef PRINT_FLOAT

/***********************************************************/
template<typename T>
void Random(T *data, size_t n, const T b = 1, const T a = 0) {
    srand(time(0));
    for(int i = 0; i < n; i ++) {
        data[i] = a + rand() % (b - a);
    }
}

#define RANDOM_INT(T)  \
    template<>  \
    void Random<T>(T *data, size_t n, const T b, const T a) {\
        std::default_random_engine seed(time(0));   \
        std::uniform_int_distribution<T> unirand(a, b); \
        for(int i = 0; i < n; i ++) data[i] = unirand(seed);    \
    }

RANDOM_INT(int)
RANDOM_INT(unsigned)
RANDOM_INT(int64_t)
RANDOM_INT(int8_t)
#undef RANDOM_INT

#define RANDOM_FLOAT(T)  \
    template<>  \
    void Random<T>(T *data, size_t n, const T b, const T a) {\
        std::default_random_engine seed(time(0));   \
        std::uniform_real_distribution<T> unirand(a, b); \
        for(int i = 0; i < n; i ++) data[i] = unirand(seed);    \
    }

RANDOM_FLOAT(float)
RANDOM_FLOAT(double)
#undef RANDOM_FLOAT

template<>
void Random<half>(half *data, size_t n, const half b, const half a) {
    std::default_random_engine seed(time(0));
    std::uniform_real_distribution<float> unirand(a, b);
    for(int i = 0; i < n; i ++) data[i] = unirand(seed);
}

/***********************************************************/