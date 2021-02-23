#include "../common.h"

#include "stdio.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <iostream>

typedef Dim3 Index3;

// Flat index with real dimension
HOSTDEVICE int FlatTensorIndex(const Index3& index, const Dim3& dims) {
  int flat_index = index[0];
  for (int i = 1; i < 3; i++) {
    flat_index = flat_index * dims[i] + index[i];
  }
  return flat_index;
}

// Convert index to tensor index with dimension.
HOSTDEVICE Index3 ConvertTensorIndex(int index, const Dim3& dims) {
  Index3 tensor_index;
  for (int i = 2; i >= 0; i--) {
    int new_index = index / dims[i];
    tensor_index[i] = index - dims[i] * new_index;
    index = new_index;
  }
  return tensor_index;
}


// Use SM to do data transfer, load a tile into SM then store out.
// All tile read and write are colascing, so can speedup memory copy
template <typename T, int NumThreads, int TileX, int TileY>
__global__ void OldTilingSwapDim1And2(const T* __restrict__ input, Dim3 input_dims,
                                   T* __restrict__ output) {
  assert(blockDim.x == NumThreads);
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);

  constexpr int BlockReadRows = NumThreads / TileY;
  constexpr int BlockWriteRows = NumThreads / TileX;

  // One extra line in the inner dimension to avoid share memory bank conflict.
  __shared__ __align__(
      alignof(T)) char share_mem_ptr[TileX * (TileY + 1) * sizeof(T)];
  typedef T(*ShareMemory)[TileY + 1];

  ShareMemory tile_sm = reinterpret_cast<ShareMemory>(share_mem_ptr);

  int x = threadIdx.x;

  Dim3 output_dims = {
      input_dims[0], input_dims[2], input_dims[1],
  };

  // Align dim to Tiles
  Dim3 tile_aligned_input_dim = {
      input_dims[0], (input_dims[1] + TileX - 1) / TileX,
      (input_dims[2] + TileY - 1) / TileY,
  };

  // Converts block idx to tile index, each block process a tile
  Index3 input_block_tile_index =
      ConvertTensorIndex(blockIdx.x, tile_aligned_input_dim);

  // Compute real index align to tile:0, 32, 64...
  Index3 block_tile_index_in_input = {
      input_block_tile_index[0], input_block_tile_index[1] * TileX,
      input_block_tile_index[2] * TileY,
  };

  // Compute block flat index against input dims.
  int input_origin_block_flat_index =
      FlatTensorIndex(block_tile_index_in_input, input_dims);

  bool full_tile = true;
  int tile_width = TileY;

  // Last row is not full.
  if (input_block_tile_index[2] == tile_aligned_input_dim[2] - 1) {
    tile_width = input_dims[2] - (tile_aligned_input_dim[2] - 1) * TileY;
    full_tile &= false;
  }

  int tile_height = TileX;

  if (input_block_tile_index[1] == tile_aligned_input_dim[1] - 1) {
    tile_height = input_dims[1] - (tile_aligned_input_dim[1] - 1) * TileX;
    full_tile &= false;
  }

  constexpr int in_effective_thread_num = NumThreads / TileY * TileY;

  if (x < in_effective_thread_num) {
    // Read a tile from input using block.
    int x_i = x / TileY;
    int x_j = x % TileY;
    int input_ind = input_origin_block_flat_index + x_i * input_dims[2] + x_j;
    int input_inc = BlockReadRows * input_dims[2];

    if (full_tile) {
#pragma unroll
      for (int ind_i = x_i; ind_i < (TileX); ind_i += BlockReadRows) {
        tile_sm[ind_i][x_j] = input[input_ind];
        input_ind += input_inc;
      }
    } else {
      if (x_j < tile_width) {
#pragma unroll
        for (int ind_i = x_i; ind_i < (tile_height); ind_i += BlockReadRows) {
          tile_sm[ind_i][x_j] = input[input_ind];
          input_ind += input_inc;
        }
      }
    }
  }

  __syncthreads();

  // Store sm value back to out
  Index3 output_block_tile_index = {
      input_block_tile_index[0], input_block_tile_index[2],
      input_block_tile_index[1],
  };

  Index3 block_tile_index_in_output = {
      output_block_tile_index[0], output_block_tile_index[1] * TileY,
      output_block_tile_index[2] * TileX,
  };

  int output_origin_block_flat_index =
      FlatTensorIndex(block_tile_index_in_output, output_dims);

  constexpr int out_effective_thread_num = NumThreads / TileX * TileX;

  if (x < out_effective_thread_num) {
    int x_i = x / TileX;
    int x_j = x % TileX;
    int output_ind =
        output_origin_block_flat_index + x_i * output_dims[2] + x_j;
    int output_inc = BlockWriteRows * output_dims[2];

    if (full_tile) {
#pragma unroll
      for (int ind_i = x_i; ind_i < (TileY); ind_i += BlockWriteRows) {
        output[output_ind] = tile_sm[x_j][ind_i];
        output_ind += output_inc;
      }
    } else {
      if (x_j < tile_height) {
#pragma unroll
        for (int ind_i = x_i; ind_i < (tile_width); ind_i += BlockWriteRows) {
          output[output_ind] = tile_sm[x_j][ind_i];
          output_ind += output_inc;
        }
      }
    }
  }
}
/*
template<typename T, int BlockDimX, int BlockDimY, int TileX, int TileY, int PadSize>
__global__ void NewTilingSwapDim1And2(const T* __restrict__ input, Dim3 input_dims,
                                   T* __restrict__ output) {
  static_assert(PadSize >= 0);
  static_assert(BlockDimX <= TileX);
  static_assert(BlockDimY <= TileY);

  // Each block transpose a tile
  // Get tile start index
  const int tile_rid = blockIdx.y;
  const int tile_cid = blockIdx.x;

  const int rid = threadIdx.y;
  const int cid = threadIdx.x;

  const int BlockTileX = (input_dims[1] + TileX - 1) / TileX;
  const int BlockTileY = (input_dims[2] + TileY - 1) / TileY;

  constexpr int TileStrideX = (TileX + BlockDimX - 1) / BlockDimX;
  constexpr int TileStrideY = (TileY + BlockDimY - 1) / BlockDimY;

  constexpr int PadTileY = TileY + PadSize;
  __shared__ T tile_sm[TileX * PadTileY];

  // Read a tile from input to shared memory
  int in_tile_index = tile_rid / BlockTileX * input_dims[1] * input_dims[2]
                    + tile_rid % BlockTileX * TileX * input_dims[2]
                    + tile_cid * TileY;
  const T __restrict__ *in_tile_beg = input + in_tile_index;
  const int in_tile_stride = input_dims[2];

  bool full_tile = true;
  int RealTileX = TileX, RealTileY = TileY;
  if(tile_rid % BlockTileX == BlockTileX - 1) {
    full_tile = false;
    RealTileX = input_dims[1] - (BlockTileX - 1) * TileX;
  }
  if(tile_cid == BlockTileY - 1) {
    full_tile = false;
    RealTileY = input_dims[2] - tile_cid * TileY;
  }

  if(full_tile) {
#pragma unroll
    for(int i = rid; i < TileX; i += TileStrideX) {
#pragma unroll
      for(int j = cid; j < TileY; j += TileStrideY) {
        tile_sm[i * PadTileY + j] = in_tile_beg[i * in_tile_stride + j];
      }
    }
  } else {
#pragma unroll
    for(int i = rid; i < RealTileX; i += TileStrideX) {
#pragma unroll
      for(int j = cid; j < RealTileY; j += TileStrideY) {
        tile_sm[i * PadTileY + j] = in_tile_beg[i * in_tile_stride + j];
      }
    }
  }
  __syncthreads();

  // Write tile from shared memory to output
  int out_tile_index = tile_rid / BlockTileX * input_dims[1] * input_dims[2]
                     + tile_rid % BlockTileX * TileX * input_dims[2]
                     + tile_cid * TileY;
  T __restrict__ *out_tile_beg = output + out_tile_index;
  const int out_tile_stride = input_dims[1];

  if(full_tile) {
#pragma unroll
    for(int i = rid; i < TileY; i += TileStrideY) {
#pragma unroll
      for(int j = cid; j < TileX; j += TileStrideX) {
        out_tile_beg[i * out_tile_stride + j] = tile_sm[j * PadTileY + i];
      }
    }
  } else {
#pragma unroll
    for(int i = rid; i < RealTileY; i += TileStrideY) {
#pragma unroll
      for(int j = cid; j < RealTileX; j += TileStrideX) {
        out_tile_beg[i * out_tile_stride + j] = tile_sm[j * PadTileY + i];
      }
    }
  }
}
*/

template<typename T, int BlockDimX, int BlockDimY, int TileDim, int PadSize>
__global__ void NewTilingSwapDim1And2(const T* __restrict__ input, Dim3 input_dims,
                                   T* __restrict__ output) {
  static_assert(BlockDimX == TileDim);
  static_assert(PadSize >= 0);
  __shared__ T tile_sm[TileDim][TileDim + PadSize];

  const int width = input_dims[2], height = input_dims[1];

  const int tile_num = (input_dims[1] + TileDim - 1) / TileDim;

  int dim0_id = blockIdx.x / tile_num;
  int block_y = blockIdx.x % tile_num;

  int base = dim0_id * width * height;

  int bidx, bidy;
  if(height == width) {
    bidy = blockIdx.y;
    bidx = (blockIdx.y + block_y) % gridDim.y;
  } else {
    int bid = blockIdx.y + gridDim.y * block_y;
    bidy = bid % tile_num;
    bidx = ((bid / tile_num) + bidy) % gridDim.y;
  }

  int xi = bidx * TileDim + threadIdx.x;
  int yi = bidy * TileDim + threadIdx.y;
  int ini = xi + yi * width + base;

  if(xi > width && yi > height) return;

  bool full_tile = true;
  int real_width = width < TileDim ? width : width - bidx * TileDim;
  int real_height = height < TileDim ? height : height - bidy * TileDim;

  if(real_width < TileDim) full_tile = false;
  else real_width = TileDim;
  if(real_height < TileDim) full_tile = false;
  else real_height = TileDim;

  if(full_tile) {
#pragma unroll
    for(int i = 0; i < TileDim; i += BlockDimY) {
      tile_sm[threadIdx.y + i][threadIdx.x] = input[ini + i * width];
    }
  } else {
    if(threadIdx.x < real_width && threadIdx.y < real_height) {
#pragma unroll
      for(int i = 0; i < real_height; i += BlockDimY) {
        tile_sm[threadIdx.y + i][threadIdx.x] = input[ini + i * width];
      }
    }
  }

  __syncthreads();

  xi = bidy * TileDim + threadIdx.x;
  yi = bidx * TileDim + threadIdx.y;
  int outi = xi + yi * height + base;

  if(full_tile) {
#pragma unroll
    for(int i = 0; i < TileDim; i += BlockDimY) {
      output[outi + i * height] = tile_sm[threadIdx.x][threadIdx.y + i];
    }
  } else {
    if(threadIdx.x < real_height && threadIdx.y < real_width) {
      const int width_bound = real_width - threadIdx.y;
#pragma unroll
      for(int i = 0; i < width_bound; i += BlockDimY) {
        output[outi + i * height] = tile_sm[threadIdx.x][threadIdx.y + i];
      }
    }
  }
}

template<typename T>
int TestTilingSwapDim1And2(CUDAStream &context, const Dim3 in_dims,
                           AllocHost &in_h, AllocDevice &in_d,
                           AllocDevice &out_old, AllocDevice &out_new) {
  size_t n = GetSize(in_dims);

  auto clock = TimeOfKernel::get(context);

  // initial data
  T *inh_ptr = in_h.ptr<T>();
  Random(inh_ptr, n);
  in_d.CopyFrom(in_h);

  T *input = in_d.ptr<T>();
  T *old_ptr = out_old.ptr<T>();
  T *new_ptr = out_new.ptr<T>();

/*******************************************************/
  // Old Swap kernel
  constexpr int kTileSize = 32;
  constexpr int kNumThreads = 256;

  Dim3 in_tiles_dims = {
    in_dims[0],
    (in_dims[1] + kTileSize - 1) / kTileSize,
    (in_dims[2] + kTileSize - 1) / kTileSize
  };
  int total_tiles_count = GetSize(in_tiles_dims);

  // initial kernel
  OldTilingSwapDim1And2<T, kNumThreads, kTileSize, kTileSize>
        <<<total_tiles_count, kNumThreads, 0, context.stream()>>>(
        input, in_dims, old_ptr);

  clock->start();
  for(int i = 0; i < 10; i ++)
  OldTilingSwapDim1And2<T, kNumThreads, kTileSize, kTileSize>
        <<<total_tiles_count, kNumThreads, 0, context.stream()>>>(
        input, in_dims, old_ptr);
  auto old_time = clock->stop();

/*******************************************************/
  // New Swap kernel
  constexpr int kTileDim = 64;
  constexpr int kBlockDimX = 64;
  constexpr int kBlockDimY = 8;
  constexpr int kPadSize = 2;

  dim3 threads(kBlockDimX, kBlockDimY);

  Dim3 new_tiles_dims = {
    in_dims[0],
    (in_dims[1] + kTileDim - 1) / kTileDim,
    (in_dims[2] + kTileDim - 1) / kTileDim
  };
  dim3 grids(new_tiles_dims[0] * new_tiles_dims[1], new_tiles_dims[2]);

  clock->start();
  for(int i = 0; i < 10; i ++)
  NewTilingSwapDim1And2<T, kBlockDimX, kBlockDimY, kTileDim, kPadSize>
        <<<grids, threads, 0, context.stream()>>>(
        input, in_dims, new_ptr);
  auto new_time = clock->stop();
  printf("Old time %f vs New time %f\n", old_time, new_time);

  auto err = context.sync();
  if(err != "") {
    fprintf(stderr, "[%d, %d, %d] CUDA ERROR: %s\n",
                    in_dims[0], in_dims[1], in_dims[2], err);
    return CUDA_FAILED;
  }
/*******************************************************/

  bool check_res = out_old.CheckSame(out_new);
  if(!check_res) {
    fprintf(stderr, "[%d, %d, %d] CHECK FAILED\n", 
                    in_dims[0], in_dims[1], in_dims[2]);

    if(n <= 2000) {
      fprintf(stderr, "Input Value:\n");
      in_h.Print<T>(in_dims[0], in_dims[1], in_dims[2]);
      fprintf(stderr, "Old Result:\n");
      out_old.Print<T>(in_dims[0], in_dims[2], in_dims[1]);
      fprintf(stderr, "New Result:\n");
      out_new.Print<T>(in_dims[0], in_dims[2], in_dims[1]);
    }
    return CHECK_FAILED;
  } else {
    printf("[%d, %d, %d] SUCCESS\n", 
           in_dims[0], in_dims[1], in_dims[2]);
  }
  return SUCCESS;
}

int main() {
  CUDAStream context;

  typedef half T;

  do {
    size_t x = 1, y = 100, z = 100;
    //printf("Input dims: ");
    //std::cin >> x >> y >> z;

    Dim3 dims = {x, y, z};
    size_t n = GetSize(dims);

    AllocHost in_h(n * sizeof(T), context);
    AllocDevice in_d(n * sizeof(T), context);
    AllocDevice out_old(n * sizeof(T), context);
    AllocDevice out_new(n * sizeof(T), context);

    if(TestTilingSwapDim1And2<T>(context, dims, in_h, in_d, out_old, out_new)
        != SUCCESS) return 1;
  } while(false);

  return 0;
}