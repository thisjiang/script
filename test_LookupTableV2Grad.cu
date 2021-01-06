#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "stdio.h"
#include "time.h"
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

template<typename TYPE>
float TimeOfKernel(TYPE *table, int64_t *ids, TYPE *output, 
                    int N, int K, int D, cudaStream_t &context) {
    dim3 threads(128, 8);
    dim3 grids(8, 1);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, context);
    LookupTableV2Grad<TYPE, 128, 8, 8><<<grids, threads, 0, context>>>(
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

double MaxError(float *data_f, half *data_h, int64_t size) {
    double maxerr = 0.0, err = 0.0;
    for(int i = 0; i < size; i ++) {
        err = fabs(data_f[i] - __half2float(data_h[i]));
        if(err > maxerr) maxerr = err;
    }
    return maxerr;
}

void TestKernel(int N, int K, int D, cudaStream_t &context) {
    //float
    float *table_d_32 = MallocDevice<float>(N * D);
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

    float time_fp32 = TimeOfKernel(table_d_32, ids_d, out_d_32, N, K, D, context);

    float *table_h_32 = MallocHost<float>(N * D);
    Device2Host(table_h_32, table_d_32, N * D, context);
    
    //half
    half *table_d_16 = MallocDevice<half>(N * D);
    half *out_h_16 = MallocHost<half>(K * D), *out_d_16 = MallocDevice<half>(K * D);

    SetZero(table_d_16, N * D, context);
    for(int i = 0; i < K * D; i ++) {
        out_h_16[i] = __float2half(out_h_32[i]);
    }
    Host2Device(out_d_16, out_h_16, K * D, context);

    float time_fp16 = TimeOfKernel(table_d_16, ids_d, out_d_16, N, K, D, context);

    half *table_h_16 = MallocHost<half>(N * D);
    Device2Host(table_h_16, table_d_16, N * D, context);

    //check result
    double maxerr_init = MaxError(out_h_32, out_h_16, K * D);
    double maxerr = MaxError(table_h_32, table_h_16, N * D);

    //print info
    printf("fp32 time %fms vs fp16 time %fms -- error:%f while initial %f\n", time_fp32, time_fp16, maxerr, maxerr_init);

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
    N = 1000000;

    srand((unsigned)time(NULL));
    cudaStream_t context;
    cudaStreamCreate(&context);

    TestKernel(1024, 1024, 1024, context);
    /*
    for(D = 64; D <= 1024; D <<= 2) {
        for(K = 1; K <= 1000000; K *= 100) {
            printf("N:%d D:%d K:%d\n", N, D, K);
            TestKernel(N, K, D, context);
            printf("\n");
        }
    }
    */

    cudaStreamDestroy(context);
    return 0;
}