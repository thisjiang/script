cmake_minimum_required(VERSION 3.10)

project(CUDAScript LANGUAGES CXX CUDA)

# Check C++ version
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)

if(NOT COMPILER_SUPPORTS_CXX17)
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_FLAGS} has no c++17 support.")
endif()

set(CMAKE_CXX_STANDARD 17 CACHE STRING "Default C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Require C++ standard")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# Check CUDA version
include(CheckLanguage)
check_language(CUDA)

# Check python version
option(PY_VERSION       "Compile PaddlePaddle with python3 support"     ${PY_VERSION})
# PY_VERSION
if(NOT PY_VERSION)
  set(PY_VERSION 3.6)
endif()
set(PYBIND11_PYTHON_VERSION ${PY_VERSION})


# set install path
if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
  message(STATUS "Setting default CMAKE_INSTALL_PREFIX path to ${CMAKE_BINARY_DIR}/install")
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE STRING "The path to use for make install" FORCE)
endif()

# add comon cmake file
include(cmake/main.cmake)
