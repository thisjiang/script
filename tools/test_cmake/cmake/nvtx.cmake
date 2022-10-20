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

if((NOT WITH_CUDA) OR WIN32 OR APPLE)
  SET(NVTX_FOUND OFF)
  RETURN()
endif()

SET(NVTX_ROOT "/usr" CACHE PATH "NVTX ROOT")
FIND_PATH(NVTX_INCLUDE_DIR nvToolsExt.h
  PATHS ${NVTX_ROOT} ${NVTX_ROOT}/include
  $ENV{NVTX_ROOT} $ENV{NVTX_ROOT}/include ${CUDA_TOOLKIT_INCLUDE}
  NO_DEFAULT_PATH
  )

GET_FILENAME_COMPONENT(__libpath_hint ${CUDA_CUDART_LIBRARY} PATH)

SET(TARGET_ARCH "x86_64")
if(NOT ${CMAKE_SYSTEM_PROCESSOR})
  SET(TARGET_ARCH ${CMAKE_SYSTEM_PROCESSOR})
endif()

LIST(APPEND NVTX_CHECK_LIBRARY_DIRS
    ${NVTX_ROOT}
    ${NVTX_ROOT}/lib64
    ${NVTX_ROOT}/lib
    ${NVTX_ROOT}/lib/${TARGET_ARCH}-linux-gnu
    $ENV{NVTX_ROOT}
    $ENV{NVTX_ROOT}/lib64
    $ENV{NVTX_ROOT}/lib
    ${CUDA_TOOLKIT_ROOT_DIR}
    ${CUDA_TOOLKIT_ROOT_DIR}/targets/${TARGET_ARCH}-linux/lib)

FIND_LIBRARY(CUDA_NVTX_LIB NAMES libnvToolsExt.so
    PATHS ${NVTX_CHECK_LIBRARY_DIRS} ${NVTX_INCLUDE_DIR} ${__libpath_hint}
    NO_DEFAULT_PATH
    DOC "Path to the NVTX library.")

if(NVTX_INCLUDE_DIR AND CUDA_NVTX_LIB)
  SET(NVTX_FOUND ON)
  message(STATUS "NVTX founded at: " ${CUDA_NVTX_LIB})
ELSE()
  SET(NVTX_FOUND OFF)
endif()

if(NVTX_FOUND)
  INCLUDE_DIRECTORIES(${NVTX_INCLUDE_DIR})
  LIST(APPEND CUDA_EXTERN_LIBRARY ${CUDA_NVTX_LIB})
endif()
