#ifndef SCRIPT_COMMON_FUNC_H
#define SCRIPT_COMMON_FUNC_H

#include "stdio.h"
#include "time.h"

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <array>
#include <initializer_list>

#include "cub/cub/cub.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "common_math.h"

/***********************************************************/
#define SUCCESS 0
#define CUDA_FAILED -1
#define CUDNN_FAILED -2
#define CHECK_FAILED 1

static inline const char* GetErrorString(const int status) {
    switch(status) {
        case SUCCESS: return "SUCCESS";
        case CUDA_FAILED: return "Cuda kernel run failed";
        case CUDNN_FAILED: return "Cudnn run failed";
        case CHECK_FAILED: return "Check Result failed";
    }
    return "Unsupported ERROR";
}

/***********************************************************/

const char* EMPTY_STRING = "";

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

/***********************************************************/
template<typename T>
T MaxErrorHost(const T *a, const T *b, const size_t n) {
    T maxerr = 0.0, err = 0.0;
    for(int i = 0; i < n; i ++) {
        err = Abs(a[i] - b[i]);
        if(err > maxerr) maxerr = err;
    }
    return maxerr;
}

template<typename T, int BLOCKDIM>
__global__ void KeMaxError(const T *a, const T *b, const size_t n, 
                           T *block_max) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_in_block = threadIdx.x;
  __shared__ T s_data[BLOCKDIM];

  T max_err(0);
  for(int i = tid; i < n; i += blockDim.x * gridDim.x) {
      T err = Abs(a[i] - b[i]);
      if(max_err < err) max_err = err;
  }
  s_data[tid_in_block] = max_err;
  __syncthreads();

  for(int k = 0; k < BLOCKDIM; k ++) {
      if(max_err < s_data[k]) max_err = s_data[k];
  }

  block_max[blockIdx.x] = max_err;
}

template<typename T, int BLOCKDIM>
T KeMaxErrorDevice(const T *a, const T *b, const size_t n,
                cudaStream_t &stream) {
    const int threads = BLOCKDIM;
    const int grids = (n + threads - 1) / threads;

    T *block_max;
    cudaMalloc(&block_max, (grids + 1) * sizeof(T));
    T *block_err;
    cudaMallocHost(&block_err, (grids + 1) * sizeof(T));

    void *tmp_mem = nullptr;
    size_t tmp_size = 0;
    cub::DeviceReduce::Max(tmp_mem, tmp_size, block_max, 
                            block_max + grids, grids, stream);
    cudaMalloc(&tmp_mem, tmp_size);

    cudaMemsetAsync(block_max, 0, (grids + 1) * sizeof(T), stream);
    KeMaxError<T, BLOCKDIM><<<grids, threads, 0, stream>>>
                                (a, b, n, block_max);
    cub::DeviceReduce::Max(tmp_mem, tmp_size, block_max, 
                           block_max + grids, grids, stream);
    
    cudaMemcpyAsync(block_err, block_max, (grids + 1) * sizeof(T), 
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    T max_err = block_err[grids];

    cudaFree(block_max); // implict sync
    cudaFree(tmp_mem);
    cudaFreeHost(block_err);

    return max_err;
}

template<typename T>
T MaxErrorDevice(const T *a, const T *b, const size_t n,
                cudaStream_t &stream) {
  if(n < 32) return KeMaxErrorDevice<T, 1>(a, b, n, stream);
  else if(n < 64) return KeMaxErrorDevice<T, 32>(a, b, n, stream);
  else if(n < 128) return KeMaxErrorDevice<T, 64>(a, b, n, stream);
  else if(n < 256) return KeMaxErrorDevice<T, 128>(a, b, n, stream);
  else if(n < 512) return KeMaxErrorDevice<T, 256>(a, b, n, stream);
  else if(n < 1024) return KeMaxErrorDevice<T, 512>(a, b, n, stream);
  else return KeMaxErrorDevice<T, 1024>(a, b, n, stream);
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
    KeCheckSameKernel<T><<<grids, threads, 0, stream>>>(a, b, n, err_d);
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
//PRINT_LONG(long long)
#undef PRINT_LONG

#define PRINT_FLOAT(T)                        \
    template<> void print<T>(T val) {printf("%f", type2type<T, float>(val));} \
    template<> void fprint<T>(T val) {fprintf(stderr, "%f", type2type<T, float>(val));}

PRINT_FLOAT(float)
PRINT_FLOAT(float16)
PRINT_FLOAT(double)

#undef PRINT_FLOAT

template<> void print<dim3>(dim3 val) {printf("[%d, %d, %d]\n", val.x, val.y, val.z);}

/***********************************************************/

template<typename T>
void Print(const T *data, const int row, const int col) {
    printf("[%d, %d]\n", row, col);
    for(int i = 0; i < row; i ++) {
        for(int j = 0; j < col; j ++) {
            print(data[i * col + j]);
            printf(" ");
        }
        printf("\n");
    }
}

template<typename T>
void Print(const T *data, const int num, const int row, const int col) {
    printf("[%d, %d, %d]\n", num, row, col);
    const int stride = row * col;
    for(int k = 0; k < num; k ++) {
        printf("[%d]\n", k);
        for(int i = 0; i < row; i ++) {
            for(int j = 0; j < col; j ++) {
                print(data[k * stride + i * col + j]);
                printf(" ");
            }
            printf("\n");
        }
    }
}

template<typename T>
void Print(const T *data, const Dim3 &dims) {
    Print(data, dims[0], dims[1], dims[2]);
}

template<typename T, size_t D>
void Print(const T *data, const std::array<int, D> &dims) {
    if(D == 1) {
        Print(data, 1, dims[0]);
        return;
    } else if(D == 2) {
        Print(data, dims[0], dims[1]);
        return;
    } else if(D == 3) {
        Print(data, dims[0], dims[1], dims[2]);
        return;
    }

    printf("[");
    for(auto num : dims) printf(" %d,", num);
    printf("]\n");

    std::array<int, D> stride;
    stride[D - 1] = 1;
    for(int i = D - 2; i >= 0; i --)
        stride[i] = dims[i + 1] * stride[i + 1];

    int len = 1;
    for(int num : dims) len *= num;
    for(int i = 0; i < len; i ++) {
        if(i % stride[D - 1] == 0) printf("\n");
        if(i % stride[D - 2] == 0) {
           printf("[");
           for(int j = 0; j < D - 2; j ++) printf(" %d,", len / stride[j]);
           printf("]\n");
        }
        print(data[i]);
        printf(" ");
    }
}


/***********************************************************/
template<typename T>
void Random(T *data, size_t n, const T b, const T a = 0) {
    srand(time(0));
    for(int i = 0; i < n; i ++) {
        data[i] = a + rand() % (b - a);
    }
}

template<typename T>
void Random(T *data, size_t n) {
    Random<T>(data, n, static_cast<T>(1), static_cast<T>(0));
}

#define RANDOM_INT(T)  \
    template<>  \
    void Random<T>(T *data, size_t n, const T b, const T a) {\
        std::default_random_engine seed(time(0));   \
        std::uniform_int_distribution<T> unirand(a, b); \
        for(int i = 0; i < n; i ++) data[i] = unirand(seed);    \
    }   \
    template<> void Random(T *data, size_t n) { \
        Random<T>(data, n, type2type<int, T>(INT_MAX), \
                  type2type<int, T>(INT_MIN)); \
    }

RANDOM_INT(int)
RANDOM_INT(unsigned)
RANDOM_INT(int64_t)
// RANDOM_INT(int8_t)
#undef RANDOM_INT

#define RANDOM_FLOAT(T)  \
    template<>  \
    void Random<T>(T *data, size_t n, const T b, const T a) {\
        std::default_random_engine seed(time(0));   \
        std::uniform_real_distribution<T> unirand(a, b); \
        for(int i = 0; i < n; i ++) data[i] = unirand(seed);    \
    }   \
    template<> void Random(T *data, size_t n) { \
        Random<T>(data, n, type2type<float, T>(10.0f), \
                  type2type<float, T>(-10.0f)); \
    }

RANDOM_FLOAT(float)
RANDOM_FLOAT(double)
#undef RANDOM_FLOAT

template<>
void Random<float16>(float16 *data, size_t n, const float16 b, const float16 a) {
    std::default_random_engine seed(time(0));
    std::uniform_real_distribution<float> unirand(a, b);
    for(int i = 0; i < n; i ++) data[i] = type2type<float, float16>(unirand(seed));
}
template<> void Random<float16>(float16 *data, size_t n) {
    Random<float16>(data, n, type2type<float, float16>(1.0f), 
                 type2type<float, float16>(-1.0f));
}

/***********************************************************/

static HOSTDEVICE size_t GetSize(const dim3 &dims) {
    return dims.x * dims.y * dims.z;
}

static HOSTDEVICE size_t GetSize(const Dim3 &dims) {
    return static_cast<size_t>(dims[0] * dims[1] * dims[2]);
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
static inline const std::string ToString(const dim3 &dims) {
    std::string res = "[" + std::to_string(dims.x) + ", ";
    res.append(std::to_string(dims.y) + ", ");
    res.append(std::to_string(dims.z) + "]");
    return res;
}

static inline const std::string ToString(const Dim3 &dims) {
    std::string res = "[" + std::to_string(dims[0]) + ", ";
    res.append(std::to_string(dims[1]) + ", ");
    res.append(std::to_string(dims[2]) + "]");
    return res;
}

template<typename T>
static inline const std::string ToString(const std::vector<T> &dims) {
    std::string res = "[";
    for(auto d : dims) res.append(std::to_string(d) + ", ");
    res.push_back(']');
    return res;
}

template<typename T, size_t D>
static inline const std::string ToString(const std::array<T, D> &dims) {
    std::string res = "[";
    for(auto d : dims) res.append(std::to_string(d) + ", ");
    res.push_back(']');
    return res;
}

template<typename T>
static inline const std::string ToString(const T *dims, int n) {
    std::string res = "[";
    for(int i = 0; i < n; i ++) res.append(std::to_string(dims[i]) + ", ");
    res.push_back(']');
    return res;
}

/***********************************************************/

#endif // SCRIPT_COMMON_FUNC_H