// Copyright (c) 2021 jiangcheng05 Authors. All Rights Reserved.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// simple version : coalesced & no bank conflict, but need m * n <= 1024
__global__ void KeMatMulSimple(float *A, float *B, float *C, int m, int k,
                               int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    float res = 0.0f;
    for (int i = 0; i < k; ++i) {
      res += A[row * k + i] * B[i * n + col];
    }
    C[row * n + col] = res;
  }
}

// shared memory : bank conflict
__global__ void KeMatrixMultiplyShared(float *A, float *B, float *C, int m,
                                       int k, int n) {
  int row = threadIdx.y, col = threadIdx.x;
  if (row < m && col < n) {
    __shared__ float s_a[TILE][TILE];
    __shared__ float s_b[TILE][TILE];

    for (int i = 0; i < k; i++) {
      s_a[row][i] = A[row][i];
      s_b[row][i] = B[row][i];
    }

    __syncthreads();

    float res = 0.0f;
    for (int i = 0; i < k; i++) {
      res += s_a[row][i] * s_b[i][col];
    }
    C[row][col] = res;
  }
}

// normal version
__global__ void KeMatrixMultiplyNormal(float *A, float *B, float *C, int m,
                                       int k, int n) {
  int row = threadIdx.y, col = threadIdx.x;
  if (row < m && col < n) {
    __shared__ float s_a[TILE][TILE];
    __shared__ float s_b[TILE][TILE + 1];

    for (int i = 0; i < k; i++) {
      s_a[row][i] = A[row][i];
      s_b[row][(i + row) % TILE] = B[row][i];
    }

    __syncthreads();

    float res = 0.0f;
    for (int i = 0; i < k; i++) {
      res += s_a[row][i] * s_b[i][(i + col) % TILE];
    }
    C[row][col] = res;
  }
}

// shuffle version
__device__ __forceinline__ float KeReduceSum(float value) {
  float res = 0.0f;
  for (int i = 16; i > 0; i /= 2) {
    res += __shfl_xor_sync(0xffffffff, value, i, 32);
  }
  return res;
}

__global__ void KeMatrixMultiplyNormal(float *A, float *B, float *C, int m,
                                       int k, int n) {
  int row = threadIdx.y, col = threadIdx.x;
  if (row < m && col < n) {
    __shared__ float v_a = A[row][col];

    float res = 0.0f;
    for (int i = 0; i < n; i++) {
      res += KeReduceSum(v_a * B[col][i]);
      C[row][i] = res;
    }
  }
}
