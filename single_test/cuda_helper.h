#ifndef SCRIPT_CUDA_HELPER_H
#define SCRIPT_CUDA_HELPER_H

#include "cuda_runtime.h"
#include "cuda_fp16.h"

/************************************************************************/
#define __CUDA_VERSION__ (__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__)

#if defined(__CUDACC__) && !defined(__CUDA_ARCH__)
#define __CUDA_ARCH__ 800
#endif

constexpr int MAX_BLOCK_DIM = 1024;
constexpr int WARP_SIZE = 32;
constexpr int HALF_WARP = WARP_SIZE / 2;


typedef cudaStream_t gpuStream_t;

/***********************************************************/
// CUDAStream
class CUDAStream final {
public:
    CUDAStream() {
        _stream = new cudaStream_t;
        cudaStreamCreate(_stream);

        if(_max_block_size == 0) GetDeviceAttr();
    }
    ~CUDAStream() {
        cudaStreamDestroy(*_stream);
        delete _stream;
    }

    CUDAStream(const CUDAStream&) = delete;
    CUDAStream(const CUDAStream&&) = delete;

    inline cudaStream_t& stream() const {return *_stream;}

    const char* sync() {
        cudaStreamSynchronize(*_stream);
        auto err = 	cudaGetLastError();
        if(err == cudaSuccess) return EMPTY_STRING;
        return cudaGetErrorString(err);
    }

    static inline int GetMaxThreadsPerBlock() {return _max_block_size;}
    static inline const dim3& GetCUDAMaxGridDimSize() {return _max_grid_dim;}
    static inline int GetMaxSharedSize() {return _max_shared_size;} 
    static inline int GetClockRate() {return _clock_rate;}
    static inline int GetSMCount() {return _sm_count;}

private:
    static inline void GetDeviceAttr() {
        cudaDeviceGetAttribute(&_max_block_size, cudaDevAttrMaxThreadsPerBlock, 0);
        cudaDeviceGetAttribute(reinterpret_cast<int *>(&_max_grid_dim.x),
                                cudaDevAttrMaxGridDimX, 0);
        cudaDeviceGetAttribute(reinterpret_cast<int *>(&_max_grid_dim.y),
                                cudaDevAttrMaxGridDimY, 0);
        cudaDeviceGetAttribute(reinterpret_cast<int *>(&_max_grid_dim.z),
                                cudaDevAttrMaxGridDimZ, 0);
        cudaDeviceGetAttribute(&_max_shared_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
        cudaDeviceGetAttribute(&_clock_rate, cudaDevAttrClockRate, 0);
        cudaDeviceGetAttribute(&_sm_count, cudaDevAttrMultiProcessorCount, 0);
    }

    cudaStream_t *_stream;
    static int _max_block_size;
    static dim3 _max_grid_dim;
    static int _max_shared_size;
    static int _clock_rate;
    static int _sm_count;
};

int CUDAStream::_max_block_size = 0;
dim3 CUDAStream::_max_grid_dim = 0;
int CUDAStream::_max_shared_size = 0;
int CUDAStream::_clock_rate = 0;
int CUDAStream::_sm_count = 0;

/************************************************************************/
// CUDA Event
class TimeOfKernel final {
private:
    TimeOfKernel() = default;
    TimeOfKernel(TimeOfKernel&) = delete;
    TimeOfKernel& operator=(const TimeOfKernel&) = delete;

    static cudaEvent_t *_start, *_stop;
    static TimeOfKernel *instance;
    static CUDAStream *_stream;

public:
    ~TimeOfKernel() {
        if(_start) cudaEventDestroy(*_start);
        if(_stop) cudaEventDestroy(*_stop);
    }

    static TimeOfKernel* get(CUDAStream &stream) {
        if(!instance) instance = new TimeOfKernel;
        _stream = &stream;
        return instance;
    }

    static void start() {
        if(!_start) {
            _start = new cudaEvent_t;
            cudaEventCreate(_start);
        }
        if(!_stop) {
            _stop = new cudaEvent_t;
            cudaEventCreate(_stop);
        }
        cudaEventRecord(*_start, _stream->stream());
    }

    static float stop() {
        cudaEventRecord(*_stop, _stream->stream());
        cudaEventSynchronize(*_stop);

        float time_of_kernel;
        cudaEventElapsedTime(&time_of_kernel, *_start, *_stop);
        return time_of_kernel;
    }
};

TimeOfKernel* TimeOfKernel::instance = nullptr;
cudaEvent_t* TimeOfKernel::_start = nullptr;
cudaEvent_t* TimeOfKernel::_stop = nullptr;
CUDAStream* TimeOfKernel::_stream = nullptr;
/************************************************************************/

template <typename T>
__forceinline__ __device__ T warpReduceSum(T val, unsigned lane_mask) {
#if (__CUDA_ARCH__ >= 800 && __CUDA_VERSION__ >= 1100)
  val = __reduce_add_sync(lane_mask, val);
#elif (__CUDA_ARCH__ >= 350 && __CUDA_VERSION__ >= 900)
  for (int mask = HALF_WARP; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(lane_mask, val, mask, warpSize);
#else
  for (int mask = HALF_WARP; mask > 0; mask >>= 1)
    val += __shfl_xor(val, mask, warpSize);
#endif
  return val;   
}

template <typename T>
__forceinline__ __device__ T warpReduceMax(T val, unsigned lane_mask) {
#if (__CUDA_ARCH__ >= 800 && __CUDA_VERSION__ >= 1100)
  val = __reduce_max_sync(lane_mask, val);
#elif (__CUDA_ARCH__ >= 350 && __CUDA_VERSION__ >= 900)
  for (int mask = HALF_WARP; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(lane_mask, val, mask, warpSize));
#else
  for (int mask = HALF_WARP; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor(val, mask, warpSize));
#endif
  return val;
}

template <typename T>
__forceinline__ __device__ T warpReduceMin(T val, unsigned lane_mask) {
#if (__CUDA_ARCH__ >= 800 && __CUDA_VERSION__ >= 1100)
  val = __reduce_min_sync(lane_mask, val);
#elif (__CUDA_ARCH__ >= 350 && __CUDA_VERSION__ >= 900)
  for (int mask = HALF_WARP; mask > 0; mask >>= 1)
    val = min(val, __shfl_xor_sync(lane_mask, val, mask, warpSize));
#else
  for (int mask = HALF_WARP; mask > 0; mask >>= 1)
    val = min(val, __shfl_xor(val, mask, warpSize));
#endif
  return val;
}

/***********************************************************/

template <typename T>
__inline__ __device__ T blockReduceSum(T val, unsigned mask) {
  static __shared__ T shared[WARP_SIZE];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val, mask);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  // align block_span to warpSize
  int block_span = (blockDim.x + warpSize - 1) >> 5;
  val = (lane < block_span) ? shared[lane] : static_cast<T>(0.0f);
  val = warpReduceSum<T>(val, mask);

  return val;
}

template <typename T>
__inline__ __device__ T blockReduceMax(T val, unsigned mask) {
  static __shared__ T shared[WARP_SIZE];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceMax(val, mask);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  // align block_span to warpSize
  int block_span = (blockDim.x + warpSize - 1) >> 5;
  val = (lane < block_span) ? shared[lane] : -1e10f;
  val = warpReduceMax(val, mask);

  return val;
}

template <typename T>
__inline__ __device__ T blockReduceMin(T val, unsigned mask) {
  static __shared__ T shared[WARP_SIZE];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceMin(val, mask);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  // align block_span to warpSize
  int block_span = (blockDim.x + warpSize - 1) >> 5;
  val = (lane < block_span) ? shared[lane] : -1e10f;
  val = warpReduceMin(val, mask);

  return val;
}

/***********************************************************/


#endif // SCRIPT_CUDA_HELPER_H