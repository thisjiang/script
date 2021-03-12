#include "../common.h"

#include "stdio.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<typename T>
__device__ T ScanPrefixSum(T *data, const int64_t num) {
  const int tid = threadIdx.x;
  if(2 * tid - 1 > num) return 0;
  int offset = 1;
  T reduce_sum = 0;
  for(int i = num >> 1; i > 0; i >>= 1) {
    __syncthreads();
    if(tid < i) {
      int a = offset * (2 * tid + 1) - 1;
      int b = offset * (2 * tid + 2) - 1;

      data[b] += data[a];
    }
    offset *= 2;
  }
  if(tid == 0) {
    for(int i = 0; i < num; i ++) {
      printf("%d ", data[i]);
    }
    reduce_sum = data[num - 1];
    data[num - 1] = 0;
  }
  for(int i = 1; i < num; i *= 2) {
    offset >>= 1;
    __syncthreads();
    if(tid < i) {
      int a = offset * (2 * tid + 1) - 1;
      int b = offset * (2 * tid + 2) - 1;
      T tmp = data[a];
      data[a] = data[b];
      data[b] += tmp;
    }
  }
  __syncthreads();
  return reduce_sum;
}


