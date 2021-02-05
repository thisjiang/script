#include "../common.h"

#include "time.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand.h>

#include <random>
#include <set>

template<typename T>
bool CheckUnique(const T *data_ptr, int num) {
  std::set<T> unique;
  for(int i = 0; i < num; i ++) {
    if(!unique.insert(data_ptr[i]).second) return false;
  }
  return true;
}

template<typename T>
__global__ void KeShuffle(T* __restrict__ data_ptr, int num, const unsigned int *rand_index) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = num - tid - 1; i > 0; i -= blockDim.x * gridDim.x) {
    int id = rand_index[i] % i;
    atomicExch(&data_ptr[id], atomicExch(&data_ptr[i], data_ptr[id]));
  }
}

template<typename T>
int CudaShuffle(T* data_ptr, int num, unsigned int seed, CUDAStream &context) {
  curandGenerator_t engine;
  curandCreateGenerator(&engine, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetStream(engine, context.stream());
  curandSetPseudoRandomGeneratorSeed(engine, seed);

  MallocDevice<unsigned int> index(num, context);
  curandGenerate(engine, index.ptr(), num);

  int threads = std::min(num, 256);
  int grids = (num + threads - 1) / threads;
  KeShuffle<<<grids, threads, 0, context.stream()>>>(data_ptr, num, index.ptr());

  curandDestroyGenerator(engine);
  return 0;
}

template<typename T>
int TestShuffle(CUDAStream &context, AllocHost &data_h, AllocDevice &data_d, int num) {
  unsigned int seed = static_cast<unsigned int>(time(0));

  T *h_ptr = data_h.ptr<T>();
  for(int i = 0; i < num; i ++) h_ptr[i] = static_cast<T>(i);
  data_d.CopyFrom(data_h);

  T *data_ptr = data_d.ptr<T>();
  CudaShuffle(data_ptr, num, seed, context);

  data_h.CopyFrom(data_d);
  auto err = context.sync();
  if(err != "") {
    fprintf(stderr, "CUDA ERROR: %s\n", err);
    return CUDA_FAILED;
  }

  if(!CheckUnique(h_ptr, num)) {
    fprintf(stderr, "Check Result Error\n");

    if(num > 200) return CHECK_FAILED;
    data_h.Print<T>(1, num);
  }

  return SUCCESS;
} 

int main() {
  CUDAStream context;

  int num = 100;
  AllocHost data_h(num * sizeof(int), context);
  AllocDevice data_d(num * sizeof(int), context);
  
  TestShuffle<int>(context, data_h, data_d, num);

  return 0;
}
