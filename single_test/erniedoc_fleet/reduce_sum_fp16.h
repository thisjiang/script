// C system file
#include "stdio.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
// C++ system file

// Library file
#include "../common.h"

constexpr int LOOPNUM = 100;

/*
template <typename Tx, typename Ty, typename TransformOp, typename ReduceOp>
typename std::enable_if<!std::is_same<Tx, paddle::platform::float16>::value, void>::type
        LaunchCubReduceKernel(
    const Tx* x_data, Ty* y_data, const platform::Place& place,
    const ReduceOp& reducer, const TransformOp& transformer, const Ty& init,
    int reduce_num, gpuStream_t stream) {
  cub::TransformInputIterator<Ty, TransformOp, const Tx*> trans_x(
    x_data, transformer);
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, trans_x, y_data,
                            reduce_num, reducer, init, stream);
  framework::Tensor tmp;
  auto* temp_storage = tmp.mutable_data<uint8_t>(
      framework::make_ddim({static_cast<int64_t>(temp_storage_bytes)}),
      place);
  cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, trans_x, y_data,
                            reduce_num, reducer, init, stream);
}

template <typename Tx, typename Ty, typename TransformOp, typename ReduceOp>
typename std::enable_if<std::is_same<Tx, paddle::platform::float16>::value, void>::type
            LaunchCubReduceKernel(
        const Tx* x_data, Ty* y_data, const platform::Place& place,
        const ReduceOp& reducer, const TransformOp& transformer, const Ty& init,
        int reduce_num, gpuStream_t stream) {
  
}
*/

/************************************************************************/
template <typename Tx, typename Ty, typename TransformOp, typename ReduceOp>
void LaunchCubReduceKernel(
    const Tx* x_data, Ty* y_data, CUDAStream& context,
    const ReduceOp& reducer, const TransformOp& transformer, const Ty& init,
    int reduce_num, gpuStream_t stream) {
  cub::TransformInputIterator<Ty, TransformOp, const Tx*> trans_x(
    x_data, transformer);
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, trans_x, y_data,
                            reduce_num, reducer, init, stream);

  AllocDevice tmp(temp_storage_bytes, context);
  auto* temp_storage = tmp.data<void>();
  cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, trans_x, y_data,
                            reduce_num, reducer, init, stream);
}

template <typename Tx, typename Ty, typename TransformOp, typename ReduceOp>
float TimeOfCubKernel(
        const Tx* x_data, Ty* y_data, CUDAStream& context,
        const ReduceOp& reducer, const TransformOp& transformer, const Ty& init,
        int reduce_num) {
  auto clock = TimeOfKernel::get(context);

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    LaunchCubReduceKernel(
        x_data, y_data, context, reducer, transformer,
        init, reduce_num, context.stream());
  }
  float cost = clock->stop();

  return cost;
}

/************************************************************************/
// template <typename Tx, typename Ty, typename TransformOp, typename ReduceOp>
// typename std::enable_if<!std::is_same<Tx, platform::float16>::value, void>::type
//         LaunchCubReduceKernel(
//     const Tx* x_data, Ty* y_data, CUDAStream& context,
//     const ReduceOp& reducer, const TransformOp& transformer, const Ty& init,
//     int reduce_num, gpuStream_t stream) {
//   cub::TransformInputIterator<Ty, TransformOp, const Tx*> trans_x(
//     x_data, transformer);
//   size_t temp_storage_bytes = 0;
//   cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, trans_x, y_data,
//                             reduce_num, reducer, init, stream);

//   AllocDevice tmp(temp_storage_bytes, context);
//   auto* temp_storage = tmp.data<uint8_t>();
//   cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, trans_x, y_data,
//                             reduce_num, reducer, init, stream);
// }

// template <typename Tx, typename Ty, typename TransformOp, typename ReduceOp>
// typename std::enable_if<std::is_same<Tx, platform::float16>::value, void>::type
//             LaunchCubReduceKernel(
//         const Tx* x_data, Ty* y_data, CUDAStream& context,
//         const ReduceOp& reducer, const TransformOp& transformer, const Ty& init,
//         int reduce_num, gpuStream_t stream) {
  
// }

template <typename Tx, typename Ty, typename ReduceOp, typename TransformOp,
          int BlockDim>
__global__ void ReduceBlockKernel(const Tx* x, Ty* y, ReduceOp reducer,
                                  TransformOp transformer, Ty init,
                                  int reduce_num) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  typedef cub::BlockReduce<Ty, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  Ty local_data = 0;
  for(int i = thread_id; i < reduce_num; i += gridDim.x * blockDim.x) {
    local_data = static_cast<Ty>(
        reducer(local_data, static_cast<Ty>(transformer(x[i]))));
  }
  __syncthreads();

  local_data = BlockReduce(temp_storage).Reduce(local_data, reducer);

  if(threadIdx.x == 0) {
    y[blockIdx.x] = local_data;
  }
}

template <typename Tx, typename Ty, typename ReduceOp, typename TransformOp,
          int BlockDim>
void LaunchReduceKernel(const Tx* x, Ty* y, const ReduceOp& reducer,
                        const TransformOp& transformer, const Ty& init,
                        int reduce_num, gpuStream_t stream,
                        CUDAStream& context) {
  int element_per_block = BlockDim * 20;
  int block_per_grid = (reduce_num + element_per_block - 1) / element_per_block;

  MallocDevice<Ty> tmp(block_per_grid, context);
  auto* temp_storage = tmp.data();

  ReduceBlockKernel<Tx, Ty, ReduceOp, TransformOp, BlockDim>
    <<<block_per_grid, BlockDim, 0, stream>>>(
    x, temp_storage, reducer, transformer, init, reduce_num);
  ReduceBlockKernel<Ty, Ty, ReduceOp, TransformOp, BlockDim>
    <<<1, BlockDim, 0, stream>>>(
    temp_storage, y, reducer, transformer, init, block_per_grid);
}

template <typename Tx, typename Ty, typename ReduceOp, typename TransformOp>
float TimeOfReduceKernel(
        const Tx* x_data, Ty* y_data, CUDAStream& context,
        const ReduceOp& reducer, const TransformOp& transformer, const Ty& init,
        int reduce_num) {
  auto clock = TimeOfKernel::get(context);
  constexpr int BlockDim = 1024;
  clock->start();
#pragma unroll
  for(int i = 0; i < 1; i ++) {
    LaunchReduceKernel<Tx, Ty, ReduceOp, TransformOp, BlockDim>(
        x_data, y_data, reducer, transformer,
        init, reduce_num, context.stream(), context);
  }
  float cost = clock->stop();

  return cost;
}

/************************************************************************/

template <typename Tx, typename Ty, typename ReduceOp, typename TransformOp,
          int BlockDim>
__global__ void ReduceKernelGrid(const Tx* x, Ty* interim, Ty* y, ReduceOp reducer,
                                  TransformOp transformer, Ty init,
                                  int reduce_num) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  typedef cub::BlockReduce<Ty, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  Ty local_interim = 0;
  for(int i = thread_id; i < reduce_num; i += gridDim.x * blockDim.x) {
    local_interim = static_cast<Ty>(
        reducer(local_interim, static_cast<Ty>(transformer(x[i]))));
  }
  __syncthreads();

  local_interim = BlockReduce(temp_storage).Reduce(local_interim, reducer);

  if(threadIdx.x == 0) {
    interim[blockIdx.x] = local_interim;
  }

  cooperative_groups::this_grid().sync();

  if(blockIdx.x == 0) {
    Ty local_y = 0;
    for(int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
      local_y = reducer(local_y, transformer(interim[i]));
    }
    __syncthreads();

    local_y = BlockReduce(temp_storage).Reduce(local_y, reducer);

    if(threadIdx.x == 0) {
      y[0] = local_y;
    }
  }
}

template <typename Tx, typename Ty, typename ReduceOp, typename TransformOp,
          int BlockDim>
void LaunchReduceKernelGrid(const Tx* x, Ty* y, const ReduceOp& reducer,
                        const TransformOp& transformer, const Ty& init,
                        int reduce_num, gpuStream_t stream,
                        CUDAStream& context) {
  int element_per_block = BlockDim * 20;
  int block_per_grid = (reduce_num + element_per_block - 1) / element_per_block;

  MallocDevice<Ty> tmp(block_per_grid, context);
  auto* temp_storage = tmp.data();

  void **args = {
    x,
    temp_storage,
    y,
    reducer,
    transformer,
    init,
    reduce_num
  };

  cudaLaunchCooperativeKernel(
    (void*)ReduceKernelGrid<Tx, Ty, ReduceOp, TransformOp, BlockDim>,
    block_per_grid,
    BlockDim,
    args,
    0,
    stream
  );
  // ReduceKernelGrid<Tx, Ty, ReduceOp, TransformOp, BlockDim>
  //   <<<block_per_grid, BlockDim, 0, stream>>>(
  //   x, temp_storage, y, reducer, transformer, init, reduce_num);
}

template <typename Tx, typename Ty, typename ReduceOp, typename TransformOp>
float TimeOfReduceKernelGrid(
        const Tx* x_data, Ty* y_data, CUDAStream& context,
        const ReduceOp& reducer, const TransformOp& transformer, const Ty& init,
        int reduce_num) {
  auto clock = TimeOfKernel::get(context);
  constexpr int BlockDim = 1024;
  clock->start();
#pragma unroll
  for(int i = 0; i < 1; i ++) {
    LaunchReduceKernelGrid<Tx, Ty, ReduceOp, TransformOp, BlockDim>(
        x_data, y_data, reducer, transformer,
        init, reduce_num, context.stream(), context);
  }
  float cost = clock->stop();

  return cost;
}

