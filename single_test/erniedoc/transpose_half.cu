#include "../common.h"

#include "stdio.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <iostream>

constexpr int LOOPNUM=100;
typedef Dim3 Index3;

// Base Reference object
template <typename T>
__global__ void KeSimpleCopy(const T* __restrict__ input, Dim3 input_dims,
                             T* __restrict__ output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = input_dims[0] * input_dims[1] * input_dims[2];
  for(int i = index; i < size; i += gridDim.x * blockDim.x)
    output[i] = input[i];
}

template<typename T>
float TimeOfCopy(const T* input, Dim3 in_dims,
                 T* output, CUDAStream &context,
                 TimeOfKernel* clock) {
  auto n = GetSize(in_dims);
  // base copy kernel
  constexpr int base_threads = 256;
  const int base_grids = (n + base_threads - 1) / base_threads;

  // Initial kernel
  KeSimpleCopy<T>
        <<<base_grids, base_threads, 0, context.stream()>>>(
        input, in_dims, output);

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++)
  KeSimpleCopy<T>
        <<<base_grids, base_threads, 0, context.stream()>>>(
        input, in_dims, output);
  return clock->stop();
}

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

template<typename T>
float TimeOfOld(const T* input, Dim3 in_dims,
                 T* output, CUDAStream &context,
                 TimeOfKernel* clock) {
  // old transpose kernel
  Dim3 in_tiles_dims = {
    in_dims[0],
    (in_dims[1] + 32 - 1) / 32,
    (in_dims[2] + 32 - 1) / 32
  };
  int total_tiles_count = GetSize(in_tiles_dims);

  // Initial kernel
  OldTilingSwapDim1And2<T, 256, 32, 32>
        <<<total_tiles_count, 256, 0, context.stream()>>>(
        input, in_dims, output);

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++)
  OldTilingSwapDim1And2<T, 256, 32, 32>
        <<<total_tiles_count, 256, 0, context.stream()>>>(
        input, in_dims, output);
  return clock->stop();
}

template<typename T, int BlockDimX, int BlockDimY, int TileDim, int PadSize>
__global__ void KeTransposeCoalesced(const T* __restrict__ input, Dim3 input_dims,
                                   T* __restrict__ output) {
  static_assert(BlockDimX == TileDim);
  __shared__ T tile_sm[TileDim][TileDim + PadSize];

  int width = input_dims[2], height = input_dims[1];
  int bidx = blockIdx.x;
  int bidy = blockIdx.y;

  int xi = bidx * TileDim + threadIdx.x;
  int yi = bidy * TileDim + threadIdx.y;
  int ini = xi + yi * width;

  xi = bidy * TileDim + threadIdx.x;
  yi = bidx * TileDim + threadIdx.y;
  int outi = xi + yi * height;

#pragma unroll
  for(int i = 0; i < TileDim; i += BlockDimY) {
    tile_sm[threadIdx.y + i][threadIdx.x] = input[ini + i * width];
  }

  __syncthreads();
#pragma unroll
  for(int i = 0; i < TileDim; i += BlockDimY) {
    output[outi + i * height] = tile_sm[threadIdx.x][threadIdx.y + i];
  }
}

template<typename T>
float TimeOfCoalesced(const T* input, Dim3 in_dims,
                 T* output, CUDAStream &context,
                 TimeOfKernel* clock) {
  // New Swap kernel
  constexpr int kTileSize = 64;
  constexpr int kBlockDimX = 64;
  constexpr int kBlockDimY = 8;
  constexpr int kPaddingSize = 2;

  dim3 threads(kBlockDimX, kBlockDimY);

  Dim3 new_tiles_dims = {
    in_dims[0],
    (in_dims[1] + kTileSize - 1) / kTileSize,
    (in_dims[2] + kTileSize - 1) / kTileSize
  };
  dim3 grids(new_tiles_dims[0] * new_tiles_dims[1], new_tiles_dims[2]);

  KeTransposeCoalesced<T, kBlockDimX, kBlockDimY, kTileSize, 2>
        <<<grids, threads, 0, context.stream()>>>(
        input, in_dims, output);

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++)
  KeTransposeCoalesced<T, kBlockDimX, kBlockDimY, kTileSize, 2>
        <<<grids, threads, 0, context.stream()>>>(
        input, in_dims, output);
  return clock->stop();
}

template<typename T, int BlockDimX, int BlockDimY, int TileDim, int PadSize>
__global__ void KeTransposeDiagonal(const T* __restrict__ input, Dim3 input_dims,
                                   T* __restrict__ output) {
  static_assert(BlockDimX == TileDim);
  static_assert(PadSize >= 0);
  assert(BlockDimX == blockDim.x);
  assert(BlockDimY == blockDim.y);

  __shared__ T tile_sm[TileDim][TileDim + PadSize];

  const int width = input_dims[2], height = input_dims[1];

  // Matrix is divided into input_dims[0]'s sub-matrix
  // Each sub-matrix are range by dimension Y
  const int grid_x = gridDim.x;
  const int grid_y = gridDim.y / input_dims[0];
  // Sub-matrix start address
  const int base = (blockIdx.y / grid_y) * width * height;
  // For sub-matrix, real block id
  const int block_y = blockIdx.y % grid_y, block_x = blockIdx.x;

  int bidx, bidy;
  // Tile are arranged by diagonal
  // Reference to https://www.cs.colostate.edu/~cs675/MatrixTranspose.pdf P17
  if(height == width) {
    bidy = block_x;
    bidx = (block_x + block_y) % grid_x;
  } else {
    int bid = block_x + grid_x * block_y;
    bidy = bid % grid_y;
    bidx = ((bid / grid_y) + bidy) % grid_x;
  }

  // Check whether or not full tile
  bool full_tile = true;
  int real_width = width < TileDim ? width : width - bidx * TileDim;
  int real_height = height < TileDim ? height : height - bidy * TileDim;

  if(real_width < TileDim) full_tile = false;
  else real_width = TileDim;
  if(real_height < TileDim) full_tile = false;
  else real_height = TileDim;

  // Read input tile to shared memory
  // Ensure coalescing and no bank-conflict
  int xi = bidx * TileDim + threadIdx.x;
  int yi = bidy * TileDim + threadIdx.y;
  int ini = xi + yi * width + base;

  if(xi > width && yi > height) return;

  if(full_tile) {
#pragma unroll
    for(int i = 0; i < TileDim; i += BlockDimY) {
      tile_sm[threadIdx.y + i][threadIdx.x] = __ldg(&input[ini + i * width]);
    }
  } else {
    if(threadIdx.x < real_width && threadIdx.y < real_height) {
      for(int i = 0; i < real_height; i += BlockDimY) {
        tile_sm[threadIdx.y + i][threadIdx.x] = __ldg(&input[ini + i * width]);
      }
    }
  }

  __syncthreads();

  // Write shared memory tile to output
  // Ensure coalescing and no bank-conflict
  xi = bidy * TileDim + threadIdx.x;
  yi = bidx * TileDim + threadIdx.y;
  int outi = xi + yi * height + base;

  if(xi > height && yi > width) return;

  if(full_tile) {
#pragma unroll
    for(int i = 0; i < TileDim; i += BlockDimY) {
      output[outi + i * height] = tile_sm[threadIdx.x][threadIdx.y + i];
    }
  } else {
    if(threadIdx.x < real_height && threadIdx.y < real_width) {
      const int width_bound = real_width - threadIdx.y;
      for(int i = 0; i < width_bound; i += BlockDimY) {
        output[outi + i * height] = tile_sm[threadIdx.x][threadIdx.y + i];
      }
    }
  }
}

template<typename T>
float TimeOfDiagonal(const T* input, Dim3 in_dims,
                 T* output, CUDAStream &context,
                 TimeOfKernel* clock) {
  // New Swap kernel
  constexpr int kTileSize = 64;
  constexpr int kBlockDimX = 64;
  constexpr int kBlockDimY = 8;
  constexpr int kPaddingSize = 2;

  dim3 threads(kBlockDimX, kBlockDimY);

  Dim3 new_tiles_dims = {
    in_dims[0],
    (in_dims[1] + kTileSize - 1) / kTileSize,
    (in_dims[2] + kTileSize - 1) / kTileSize
  };
  dim3 grids(new_tiles_dims[2], new_tiles_dims[0] * new_tiles_dims[1]);

  KeTransposeDiagonal<T, kBlockDimX, kBlockDimY, kTileSize, kPaddingSize>
        <<<grids, threads, 0, context.stream()>>>(
        input, in_dims, output);

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++)
  KeTransposeDiagonal<T, kBlockDimX, kBlockDimY, kTileSize, kPaddingSize>
        <<<grids, threads, 0, context.stream()>>>(
        input, in_dims, output);
  return clock->stop();
}

template<typename T, int BlockDimX, int BlockDimY, int TileX, int TileY, int PadSize>
__global__ void KeTransposeDiagonalXandY(const T* __restrict__ input, Dim3 input_dims,
                                   T* __restrict__ output) {
  static_assert(BlockDimX == TileX);
  static_assert(BlockDimX >= TileY);
  static_assert(BlockDimY <= TileY);
  static_assert(PadSize >= 0);

  __shared__ __align__(alignof(T)) T tile_sm[TileY][TileX + PadSize];

  const int width = input_dims[2], height = input_dims[1];

  // Matrix is divided into input_dims[0]'s sub-matrix
  // Each sub-matrix are range by dimension Y
  const int grid_x = gridDim.x;
  const int grid_y = gridDim.y / input_dims[0];
  // Sub-matrix start address
  const int base = (blockIdx.y / grid_y) * width * height;
  // For sub-matrix, real block id
  const int block_y = blockIdx.y % grid_y, block_x = blockIdx.x;

  // Tile are arranged by diagonal
  // Reference to https://www.cs.colostate.edu/~cs675/MatrixTranspose.pdf P17
  const int bid = block_x + grid_x * block_y;
  const int bidy = bid % grid_y;
  const int bidx = ((bid / grid_y) + bidy) % grid_x;

  // Check whether or not full tile
  bool full_tile = true;
  int real_width = width < TileX ? width : width - bidx * TileX;
  int real_height = height < TileY ? height : height - bidy * TileY;

  if(real_width < TileX) full_tile = false;
  else real_width = TileX;
  if(real_height < TileY) full_tile = false;
  else real_height = TileY;

  // Read input tile to shared memory
  // Ensure coalescing and no bank-conflict
  int xi = bidx * TileX + threadIdx.x;
  int yi = bidy * TileY + threadIdx.y;
  int ini = xi + yi * width + base;


  if(full_tile) {
#pragma unroll
    for(int i = 0; i < TileY; i += BlockDimY) {
      tile_sm[threadIdx.y + i][threadIdx.x] = __ldg(&input[ini + i * width]);
    }
  } else {
    if(threadIdx.x < real_width && threadIdx.y < real_height) {
#pragma unroll
      for(int i = 0; i < real_height; i += BlockDimY) {
        tile_sm[threadIdx.y + i][threadIdx.x] = __ldg(&input[ini + i * width]);
      }
    }
  }

  __syncthreads();

  // threadIdx.x should less than TileY
  if(threadIdx.x >= real_height) return;

  // Write shared memory tile to output
  // Ensure coalescing and no bank-conflict
  xi = bidy * TileY + threadIdx.x;
  yi = bidx * TileX + threadIdx.y;
  int outi = xi + yi * height + base;

  if(full_tile) {
#pragma unroll
    for(int i = 0; i < TileX; i += BlockDimY) {
      output[outi + i * height] = tile_sm[threadIdx.x][threadIdx.y + i];
    }
  } else {
    if(threadIdx.y < real_width) {
      const int width_bound = real_width - threadIdx.y;
#pragma unroll
      for(int i = 0; i < width_bound; i += BlockDimY) {
        output[outi + i * height] = tile_sm[threadIdx.x][threadIdx.y + i];
      }
    }
  }
}

template<typename T>
float TimeOfDiagonalXandY(const T* input, Dim3 in_dims,
                 T* output, CUDAStream &context,
                 TimeOfKernel* clock) {
  constexpr int kTileX = 64;
  constexpr int kTileY = 64;
  constexpr int kBlockDimX = 64;
  constexpr int kBlockDimY = 4;
  constexpr int kPaddingSize = 2;

  dim3 threads(kBlockDimX, kBlockDimY);

  Dim3 new_tiles_dims = {
    in_dims[0],
    (in_dims[1] + kTileY - 1) / kTileY,
    (in_dims[2] + kTileX - 1) / kTileX
  };
  dim3 grids(new_tiles_dims[2], new_tiles_dims[0] * new_tiles_dims[1]);
  // Initial kernel
  KeTransposeDiagonalXandY<T, kBlockDimX, kBlockDimY, kTileX, kTileY, kPaddingSize>
        <<<grids, threads, 0, context.stream()>>>(
        input, in_dims, output);

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++)
  KeTransposeDiagonalXandY<T, kBlockDimX, kBlockDimY, kTileX, kTileY, kPaddingSize>
        <<<grids, threads, 0, context.stream()>>>(
        input, in_dims, output);
  return clock->stop();
}

template<typename T>
int TestTranspose(CUDAStream &context, const Dim3& in_dims,
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

  auto base_time = TimeOfCopy(input, in_dims, old_ptr, context, clock);
  auto old_time = TimeOfOld(input, in_dims, old_ptr, context, clock);
  //auto new_time = TimeOfCoalesced(input, in_dims, new_ptr, context, clock);
  //auto new_time = TimeOfDiagonal(input, in_dims, new_ptr, context, clock);
  auto new_time = TimeOfDiagonalXandY(input, in_dims, new_ptr, context, clock);

  printf("Base time %f vs Old time %f vs New time %f\n",
          base_time, old_time, new_time);

  auto err = context.sync();
  if(err != "") {
    fprintf(stderr, "[%d, %d, %d] CUDA ERROR: %s\n",
                    in_dims[0], in_dims[1], in_dims[2], err);
    return CUDA_FAILED;
  }

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
  srand(time(NULL));

  do {
    size_t x = 1, y = 458752, z = 48;
    //printf("Input dims: ");
    //std::cin >> x >> y >> z;

    Dim3 dims = {x, y, z};
    size_t n = GetSize(dims);

    AllocHost in_h(n * sizeof(T), context);
    AllocDevice in_d(n * sizeof(T), context);
    AllocDevice out_old(n * sizeof(T), context);
    AllocDevice out_new(n * sizeof(T), context);

    if(TestTranspose<T>(context, dims, in_h, in_d, out_old, out_new)
        != SUCCESS) return 1;
  } while(false);

  return 0;
}