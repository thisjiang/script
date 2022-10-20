# Copyright (c) 2022 thisjiang Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

find_package(CUDA 11.2 REQUIRED)

message(STATUS "CUDA founded at: " ${CUDA_TOOLKIT_ROOT_DIR})

ENABLE_LANGUAGE(CUDA)

SET(CMAKE_CUDA_STANDARD 14)
SET(CMAKE_CUDA_STANDARD_REQUIRED ON)

cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)
LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
SET(CUDA_SEPARABLE_COMPILATION ON)

FIND_LIBRARY(CUDASTUB libcuda.so HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/ REQUIRED)
FIND_LIBRARY(CUBLAS libcublas.so HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 /usr/lib REQUIRED)
FIND_LIBRARY(CUDNN libcudnn.so HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 /usr/lib REQUIRED)

LIST(APPEND CUDA_EXTERN_LIBRARY ${CUDA_NVRTC_LIB} ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY} ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/libcuda.so)
