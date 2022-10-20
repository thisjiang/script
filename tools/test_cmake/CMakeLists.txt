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

cmake_minimum_required(VERSION 3.10)

project(LearningCmake CXX C)

include(cmake/config.cmake)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})

option(WITH_TEST "Compile with Unittests" ON)
option(WITH_CUDA "Compile with CUDA" OFF)

set(THIRD_PARTY_PATH "${CMAKE_BINARY_DIR}/third_party")
set(EXTERN_LIBRARY CACHE INTERNAL "" FORCE)

include(cmake/gflags.cmake)
include(cmake/glog.cmake)

if (WITH_CUDA)
  set(CUDA_EXTERN_LIBRARY CACHE INTERNAL "" FORCE)
  include(cmake/cuda.cmake)
  include(cmake/nvtx.cmake)
endif()

if (WITH_TEST)
  set(TEST_LIBRARY CACHE INTERNAL "" FORCE)
  include(cmake/gtest.cmake)
endif()

include(cmake/utils.cmake)

add_subdirectory(src)

if (WITH_TEST)
  add_subdirectory(tests)
endif()
