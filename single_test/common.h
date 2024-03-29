

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS
#undef EIGEN_GPU_COMPILE_PHASE
#include "eigen/unsupported/Eigen/CXX11/Tensor"

#include "common_func.h"
#include "alloc_helper.h"
#include "cuda_helper.h"
#include "cudnn_helper.h"