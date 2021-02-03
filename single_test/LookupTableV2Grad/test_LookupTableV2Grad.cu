#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <vector>

#include "stdio.h"
#include "time.h"

template <typename T, int BlockDimX, int BlockDimY, int GridDimX>
__global__ void LookupTableV2GradBase(T *table, const T *output, const int64_t *ids,
                                  const int64_t N, const int64_t K,
                                  const int64_t D) {
  int idx = threadIdx.x;
  int idy = blockIdx.x + threadIdx.y * GridDimX;

  while (idy < K) {
    int64_t id = ids[idy];
    const T *out = output + idy * D;
    T *tab = table + id * D;
    for (int i = idx; i < D; i += BlockDimX) {
      atomicAdd(&tab[i], out[i]);
    }
    idy += BlockDimY * GridDimX;
  }
}
/*
template <typename T, int BlockDimX, int BlockDimY, int GridDimX>
__global__ void LookupTableV2Grad(T *table, const T *output, const int64_t *ids,
                                  const int64_t N, const int64_t K,
                                  const int64_t D) {
  int tid_of_block = threadIdx.x + threadIdx.y * BlockDimX;
  int wid_of_block = tid_of_block / 32;
  constexpr int wnum_of_block = (BlockDimX * BlockDimY + 31) / 32;

  int wid = wid_of_block + blockIdx.x * wnum_of_block;
  int tid = tid_of_block % 32;

  while(wid < D) {
    for(int i = tid; i < K; i += 32) {
      int64_t id = ids[i];
      const T *out = output + i * D;
      T *tab = table + id * D;

      __shared__ T s_out[BlockDimX * BlockDimY];
      __shared__ T s_tab[BlockDimX * BlockDimY];

      s_out[] = output + wid_of_block * D + tid;

      atomicAdd(&s_tab[tid_of_block], s_out[tid_of_block]);
    }
    wid += GridDimX * wnum_of_block;
  }
}
*/
/*
template <typename T, int BlockDimX, int BlockDimY, int GridDimX>
__global__ void LookupTableV2Grad(T *table, const T *output, const int64_t *ids,
                                  const int64_t N, const int64_t K,
                                  const int64_t D) {
  int idx = threadIdx.x;
  int idy = blockIdx.x + threadIdx.y * GridDimX;

  while (idy < K) {
    int64_t id = ids[idy];
    const T *out = output + idy * D;
    T *tab = table + id * D;
    for (int i = idx; i < D; i += BlockDimX) {
      atomicAdd(&tab[i], out[i]);
    }
    idy += BlockDimY * GridDimX;
  }
}
*/

/*
    eg: Tile = 4, for each warp each loop
    data:    out0 out1 ...  out7  out0   out1 ... out7
    row:      0     0  ...    0     1     1   ...   3
      |       |     |         |     |     |         |
      V       V     V         V     V     V         V
    tile:   tile0 tile1 ... tile7 tile0 tile1 ... tile7
      |       |     |         |     |     |         |
      V       V     V         V     V     V         V
    thread:   t0   t1   ...  t7    t8    t9   ...  t31
                       CudaAtomicAdd
      |       |     |         |     |     |         |
      V       V     V         V     V     V         V
    data:    tab0 tab1 ...  tab7  tab0   tab1 ... tab7
    row:     ids0 ids0 ...  ids0  ids1   ids1 ... ids3
  */
/*
template <typename T, int Tile>
__global__ void LookupTableV2Grad(T *table, const T *output, const int64_t *ids,
                                  const int64_t N, const int64_t K,
                                  const int64_t D) {                          
  const int tid = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
  const int warp_id = tid / 32, tid_of_warp = tid % 32;
  const int tile_num = (gridDim.x * blockDim.x * blockDim.y + Tile - 1) / Tile;
  const int tile_of_warp = 32 / Tile;  // tile_of_warp = 8
  int tile_id = warp_id * tile_of_warp + tid_of_warp % 8;

  while(tile_id < D) {
    for(int i = tid_of_warp / tile_of_warp; i < K; i += Tile) {
      int64_t id = ids[i];
      const T *out = output + i * D;
      T *tab = table + id * D;

      atomicAdd(&tab[tile_id], out[tile_id]);
    }
    tile_id += tile_num;
  }
}
*/

template <typename T, int BlockSize>
__global__ void LookupTableV2Grad(T *table, const T *output, const int64_t *ids,
                                  const int64_t N, const int64_t K,
                                  const int64_t D) {
  int tid = threadIdx.x;
  const int64_t stride = (N + gridDim.x - 1) / gridDim.x;
  const int64_t N_beg = blockIdx.x * stride;
  const int64_t N_end = N_beg + stride;

  __shared__ int64_t s_ids[BlockSize];
  for(int64_t k = 0; k < K; k += BlockSize) {
      s_ids[tid] = ids[k + tid];
    __syncthreads();

#pragma unroll
    for (int i = 0; i < BlockSize; i++) {
      int64_t id = s_ids[i];
      if(id >= N_beg && id < N_end) {
        const T *out = output + (k + i) * D;
        T *tab = table + id * D;

#pragma unroll
        for(int idx = tid; idx < D; idx += BlockSize)
          tab[idx] += out[idx];
      }
    }
  }
}

template<typename TYPE>
float TimeOfBase(TYPE *table, int64_t *ids, TYPE *output, 
                    int N, int K, int D, cudaStream_t &context) {
    dim3 threads(128, 8);
    dim3 grids(8);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, context);
    LookupTableV2GradBase<TYPE, 128, 8, 8><<<grids, threads, 0, context>>>(
        table, output, ids, N, K, D);
    cudaEventRecord(stop, context);
    cudaEventSynchronize(stop);

    float time_of_kernel;
    cudaEventElapsedTime(&time_of_kernel, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time_of_kernel;
}

template<typename TYPE>
float TimeOfTest(TYPE *table, int64_t *ids, TYPE *output, 
                    int N, int K, int D, cudaStream_t &context) {
    dim3 threads(256);
    dim3 grids(std::min(K, 16), 1);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, context);
    LookupTableV2Grad<TYPE, 256><<<grids, threads, 0, context>>>(
        table, output, ids, N, K, D);
    cudaEventRecord(stop, context);
    cudaEventSynchronize(stop);

    float time_of_kernel;
    cudaEventElapsedTime(&time_of_kernel, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time_of_kernel;
}

template<typename T>
T* MallocDevice(int64_t size) {
    T *ptr_d;
    cudaMalloc((void**)&ptr_d, size * sizeof(T));
    return ptr_d;
}
template<typename T>
T* MallocHost(int64_t size) {
    T *ptr_h;
    cudaMallocHost((void**)&ptr_h, size * sizeof(T));
    return ptr_h;
}
template<typename T>
void SetZero(T *ptr, int64_t size, cudaStream_t &context) {
    cudaMemsetAsync(ptr, 0, size * sizeof(T), context);
}

template<typename T>
void Host2Device(T *des, T *src, int64_t size, cudaStream_t &context) {
    cudaMemcpyAsync(des, src, size * sizeof(T), cudaMemcpyHostToDevice, context);
}
template<typename T>
void Device2Host(T *des, T *src, int64_t size, cudaStream_t &context) {
    cudaMemcpyAsync(des, src, size * sizeof(T), cudaMemcpyDeviceToHost, context);
}

template<typename T>
float Convert2Float(T input) {
    return static_cast<float>(input);
}

template<>
float Convert2Float<half>(half input) {
    return __half2float(input);
}
template<typename T1, typename T2>
double MaxError(T1 *data_1, T2 *data_2, int64_t size) {
    double maxerr = 0.0, err = 0.0;
    for(int i = 0; i < size; i ++) {
        err = fabs(Convert2Float<T1>(data_1[i]) - Convert2Float<T2>(data_2[i]));
        if(err > maxerr) maxerr = err;
    }
    return maxerr;
}

void TestKernel(int N, int K, int D, cudaStream_t &context) {
    //float
    float *table_d_32 = MallocDevice<float>(N * D), *table_d32_base = MallocDevice<float>(N * D);
    int64_t *ids_h = MallocHost<int64_t>(K), *ids_d = MallocDevice<int64_t>(K);
    float *out_h_32 = MallocHost<float>(K * D), *out_d_32 = MallocDevice<float>(K * D);

    SetZero(table_d_32, N * D, context);
    for(int i = 0; i < K; i ++) {
        ids_h[i] = rand() % N;
    }
    Host2Device(ids_d, ids_h, K, context);
    for(int i = 0; i < K * D; i ++) {
         out_h_32[i] = (rand() % 2000000 - 1000000.0f) / 100000.0f;
    }
    Host2Device(out_d_32, out_h_32, K * D, context);

    float time_fp32 = TimeOfTest(table_d_32, ids_d, out_d_32, N, K, D, context);
    float time_fp32_base = TimeOfBase(table_d32_base, ids_d, out_d_32, N, K, D, context);

    float *table_h_32 = MallocHost<float>(N * D), *table_h32_base = MallocHost<float>(N * D);
    Device2Host(table_h_32, table_d_32, N * D, context);
    Device2Host(table_h32_base, table_d32_base, N * D, context);
    
    //half
    half *table_d_16 = MallocDevice<half>(N * D), *table_d16_base = MallocDevice<half>(N * D);
    half *out_h_16 = MallocHost<half>(K * D), *out_d_16 = MallocDevice<half>(K * D);

    SetZero(table_d_16, N * D, context);
    for(int i = 0; i < K * D; i ++) {
        out_h_16[i] = __float2half(out_h_32[i]);
    }
    Host2Device(out_d_16, out_h_16, K * D, context);

    float time_fp16 = TimeOfTest(table_d_16, ids_d, out_d_16, N, K, D, context);
    float time_fp16_base = TimeOfBase(table_d16_base, ids_d, out_d_16, N, K, D, context);

    half *table_h_16 = MallocHost<half>(N * D), *table_h16_base = MallocHost<half>(N * D);
    Device2Host(table_h_16, table_d_16, N * D, context);
    Device2Host(table_h16_base, table_d16_base, N * D, context);

    //check result
    double maxerr_init = MaxError(out_h_32, out_h_16, K * D);
    double maxerr = MaxError(table_h_32, table_h_16, N * D);
    double maxerr_base = MaxError(table_h32_base, table_h16_base, N * D);

    //print info
    /*
    printf("fp32 test time %fms vs base time %fms\n", time_fp32, time_fp32_base);
    printf("fp16 test time %fms vs base time %fms\n", time_fp16, time_fp16_base);
    printf("Error test vs base: %f vs %f\n", maxerr, maxerr_base);
    */
   printf("%d %d %d %f %f %f %f %f %f\n", 
          N, K, D, time_fp32, time_fp32_base, time_fp16, time_fp16_base, maxerr, maxerr_base);

    cudaFree(table_d_32);
    cudaFree(table_d_16);
    cudaFree(ids_d);
    cudaFree(out_d_32);
    cudaFree(out_d_16);
    cudaFreeHost(table_h_32);
    cudaFreeHost(table_h_16);
    cudaFreeHost(ids_h);
    cudaFreeHost(out_h_32);
    cudaFreeHost(out_h_16);
}

int main() {
    int64_t N, K, D;
    N = 1024;
    K = 100;
    D = 786;

    srand((unsigned)time(NULL));
    cudaStream_t context;
    cudaStreamCreate(&context);
    printf("N K D test32-time/ms base32-time/ms test16-time/ms base16-time/ms error32 error16\n");

    //TestKernel(N, K, D, context);
    
    std::vector<int64_t> vecN, vecK, vecD;
    vecN = {1, 2, 100, 1024, 10240};
    vecK = {1, 2, 100, 1024, 10240};
    vecD = {32, 256, 768, 1024};
    for(int i = 0; i < vecD.size(); i ++) {
      for(int j = 0; j < vecK.size(); j ++) {
        for(int k = 0; k < vecN.size(); k ++) {
            TestKernel(vecN[k], vecK[j], vecD[i], context);
        }
      }
    }
    
    cudaStreamDestroy(context);
    return 0;
}