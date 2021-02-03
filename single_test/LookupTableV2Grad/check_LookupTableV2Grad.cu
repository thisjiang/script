#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <fstream>
#include <vector>

#include "stdio.h"
#include "time.h"

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

/*
    data:    out0 out1 ...  out7  out0   out1 ... out7
    row:     ids0 ids0 ...  ids0  ids1   ids1 ... ids3  
      |       |     |         |     |     |         |
      V       V     V         V     V     V         V
    tile:   tile0 tile1 ... tile7 tile0 tile1 ... tile7
      |       |     |         |     |     |         |
      V       V     V         V     V     V         V
    thread:   t0   t1   ...  t7    t8    t9   ...  t31  
    
    each tile (4 threads) atomicAdd K data
*/
/*
template <typename T, int Tile>
__global__ void LookupTableV2Grad2(T *table, const T *output, const int64_t *ids,
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
}*/
/*
template <typename T>
__global__ void LookupTableV2Grad2(T *table, const T *output, const int64_t *ids,
                                  const int64_t N, const int64_t K,
                                  const int64_t D) {
  const int stride = gridDim.x * blockDim.x * blockDim.y;
  int64_t idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;

  extern __shared__ int64_t s_ids[];
#pragma unroll
  for(int64_t i = idx; i < K; i += blockDim.x * blockDim.y) s_ids[i] = ids[i];
  __syncthreads();

  while(idx < D) {
#pragma unroll
    for(int64_t i = 0; i < K; i ++) {
      int64_t id = s_ids[i];
      const T *out = output + i * D;
      T *tab = table + id * D;

      tab[idx] += out[idx];
    }
    idx += stride;
  }
}
*/
template <typename T>
__global__ void LookupTableV2Grad2(T *table, const T *output, const int64_t *ids,
                                  const int64_t N, const int64_t K,
                                  const int64_t D, const int64_t padding_idx) {
  const int stride = gridDim.x * blockDim.x;
  int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;

  extern __shared__ int64_t s_ids[];
#pragma unroll
  for (int64_t i = threadIdx.x; i < K; i += blockDim.x) s_ids[i] = ids[i];
  __syncthreads();

  T *tab_padding = table + padding_idx * D;
  while (idx < D) {
    T padding_sum = tab_padding[idx];

#pragma unroll
    for (int64_t i = 0; i < K; i++) {
      int64_t id = s_ids[i];
      const T *out = output + i * D;
      T *tab = table + id * D;

      if(id == padding_idx)
        padding_sum += out[idx];
      else 
        tab[idx] += out[idx];
    }
    tab_padding[idx] = padding_sum;
    idx += stride;
  }
}

template <typename T>
__global__ void LookupTableV2Grad_N16(T *table, const T *output, const int64_t *ids,
                                  const int64_t N, const int64_t K,
                                  const int64_t D){
  const int stride = gridDim.x * blockDim.x;
  int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;

  extern __shared__ int64_t s_ids[];
#pragma unroll
  for (int64_t i = threadIdx.x; i < K; i += blockDim.x) s_ids[i] = ids[i];
  __syncthreads();

  while (idx < D) {
    T tab_res[16]{(T)0};
#pragma unroll
    for(int i = 0; i < N; i ++) {
      T *tab = table + i * D;
      tab_res[i] = tab[idx];
    }
#pragma unroll
    for (int64_t i = 0; i < K; i++) {
      int64_t id = s_ids[i];
      const T *out = output + i * D;
      tab_res[id] += out[idx];
    }
#pragma unroll
    for(int i = 0; i < N; i ++) {
      T *tab = table + i * D;
      tab[idx] = tab_res[i];
    }
    idx += stride;
  }
}

template<typename T>
float Convert2Float(T input) {
    return static_cast<float>(input);
}

template<>
float Convert2Float<half>(half input) {
    return __half2float(input);
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
double MaxError(T *data_1, T *data_2, int64_t size) {
    double maxerr = 0.0, err = 0.0;
    for(int i = 0; i < size; i ++) {
        err = fabs(Convert2Float<T>(data_1[i]) - Convert2Float<T>(data_2[i]));
        if(err > maxerr) maxerr = err;
    }
    return maxerr;
}
template<typename T>
void Print2File(T *data, int64_t N, int64_t D, const char *filename) {
    std::fstream file(filename, std::ios_base::out);
    for(int i = 0; i < N; i ++) {
        for(int j = 0; j < D; j ++) {
            file<<Convert2Float<T>(data[i * D + j])<< " ";
        }
        file<<std::endl;
    }
}

void TestKernel(int N, int K, int D, cudaStream_t &context) {
    dim3 threads1(128, 8);
    dim3 grids1(8, 1);

    int threads2 = std::min(D, 256);
    int grids2((D + 255) / 256);

    //float
    float *table_d32_1 = MallocDevice<float>(N * D), *table_d32_2 = MallocDevice<float>(N * D);
    int64_t *ids_h = MallocHost<int64_t>(K), *ids_d = MallocDevice<int64_t>(K);
    float *out_h_32 = MallocHost<float>(K * D), *out_d_32 = MallocDevice<float>(K * D);

    SetZero(table_d32_1, N * D, context);
    SetZero(table_d32_2, N * D, context);
    for(int i = 0; i < K; i ++) {
        ids_h[i] = rand() % N;
    }
    Host2Device(ids_d, ids_h, K, context);
    for(int i = 0; i < K * D; i ++) {
         out_h_32[i] = (rand() % 2000000 - 1000000.0f) / 100000.0f;
    }
    Host2Device(out_d_32, out_h_32, K * D, context);

    LookupTableV2Grad<float, 128, 8, 8><<<grids1, threads1, 0, context>>>(
          table_d32_1, out_d_32, ids_d, N, K, D);
    if(N <= 16) {
      LookupTableV2Grad_N16<float><<<grids2, threads2, K * sizeof(int64_t), context>>>(
          table_d32_2, out_d_32, ids_d, N, K, D);
    } else {
      LookupTableV2Grad2<float><<<grids2, threads2, K * sizeof(int64_t), context>>>(
          table_d32_2, out_d_32, ids_d, N, K, D, 0);
    }

    float *table_h32_1 = MallocHost<float>(N * D), *table_h32_2 = MallocHost<float>(N * D);
    Device2Host(table_h32_1, table_d32_1, N * D, context);
    Device2Host(table_h32_2, table_d32_2, N * D, context);
    
    //half
    half *table_d16_1 = MallocDevice<half>(N * D), *table_d16_2 = MallocDevice<half>(N * D);
    half *out_h_16 = MallocHost<half>(K * D), *out_d_16 = MallocDevice<half>(K * D);

    SetZero(table_d16_1, N * D, context);
    SetZero(table_d16_2, N * D, context);
    for(int i = 0; i < K * D; i ++) {
        out_h_16[i] = __float2half(out_h_32[i]);
    }
    Host2Device(out_d_16, out_h_16, K * D, context);

    LookupTableV2Grad<half, 128, 8, 8><<<grids1, threads1, 0, context>>>(
          table_d16_1, out_d_16, ids_d, N, K, D);
    if(N <= 16) {
      LookupTableV2Grad_N16<half><<<grids2, threads2, K * sizeof(int64_t), context>>>(
          table_d16_2, out_d_16, ids_d, N, K, D);
    } else {
      LookupTableV2Grad2<half><<<grids2, threads2, K * sizeof(int64_t), context>>>(
          table_d16_2, out_d_16, ids_d, N, K, D, 0);
    }

    half *table_h16_1 = MallocHost<half>(N * D), *table_h16_2 = MallocHost<half>(N * D);
    Device2Host(table_h16_1, table_d16_1, N * D, context);
    Device2Host(table_h16_2, table_d16_2, N * D, context);

    //check result
    cudaStreamSynchronize(context);
    double maxerr32 = MaxError(table_h32_1, table_h32_2, N * D);
    double maxerr16 = MaxError(table_h16_1, table_h16_2, N * D);

    printf("fp32 maxerr %f and fp16 maxerr %f\n", maxerr32, maxerr16);

    // out to file
    Print2File(table_h32_1, N, D, "table_fp32_1.log");
    Print2File(table_h32_2, N, D, "table_fp32_2.log");
    Print2File(table_h16_1, N, D, "table_fp16_1.log");
    Print2File(table_h16_2, N, D, "table_fp16_2.log");

    // free device memory
    cudaFree(table_d32_1);
    cudaFree(table_d32_2);
    cudaFree(table_d16_1);
    cudaFree(table_d16_2);
    cudaFree(ids_d);
    cudaFree(out_d_32);
    cudaFree(out_d_16);

    // free host memory
    cudaFreeHost(table_h32_1);
    cudaFreeHost(table_h32_2);
    cudaFreeHost(table_h16_1);
    cudaFreeHost(table_h16_2);
    cudaFreeHost(ids_h);
    cudaFreeHost(out_h_32);
    cudaFreeHost(out_h_16);
}

int main() {
    int64_t N, K, D;
    N = K = D = 1024;

    srand((unsigned)time(NULL));
    cudaStream_t context;
    cudaStreamCreate(&context);

    //TestKernel(N, K, D, context);
    std::vector<int64_t> vecN, vecK, vecD;
    vecN = {2, 64, 100, 1024, 10240};
    vecK = {2, 64, 100, 1024, 10240};
    vecD = {32, 256, 512, 768, 1024};
    for(int i = 0; i < vecN.size(); i ++) {
      for(int j = 0; j < vecK.size(); j ++) {
        for(int k = 0; k < vecD.size(); k ++) {
            printf("N:%d D:%d K:%d\n", vecN[i], vecD[i], vecK[i]);
            TestKernel(vecN[i], vecK[j], vecD[k], context);
            printf("\n");
        }
      }
    }

    cudaStreamDestroy(context);
    return 0;
}