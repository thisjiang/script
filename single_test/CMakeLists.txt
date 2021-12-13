cmake_minimum_required(VERSION 3.16)

project(CUDAScript LANGUAGES CXX CUDA)

# add global argument
option(PY_VERSION       "Compile with python3 support"     ${PY_VERSION})

include(cmake/check.cmake)

# print config value
message(STATUS "C++ compile: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA compile: ${CMAKE_CUDA_COMPILER}")
message(STATUS "Python version: ${Python3_EXECUTABLE}")


# set CUDA arguments
include(cmake/cuda.cmake)

# set python arguments
set(PYBIND11_PYTHON_VERSION ${PY_VERSION})

# set install path
if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
  message(STATUS "Setting default CMAKE_INSTALL_PREFIX path to ${CMAKE_BINARY_DIR}/install")
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE STRING "The path to use for make install" FORCE)
endif()

# add third party
SET(EXTERNAL_PROJECT_LOG_ARGS
LOG_DOWNLOAD    0     # Wrap download in script to log output
LOG_UPDATE      1     # Wrap update in script to log output
LOG_CONFIGURE   1     # Wrap configure in script to log output
LOG_BUILD       0     # Wrap build in script to log output
LOG_TEST        1     # Wrap test in script to log output
LOG_INSTALL     0     # Wrap install in script to log output
)

# some necessarily macro
MACRO(UNSET_VAR VAR_NAME)
    UNSET(${VAR_NAME} CACHE)
    UNSET(${VAR_NAME})
ENDMACRO()

# import third party library
include(ExternalProject)

set(THIRD_PARTY_PATH ${CMAKE_CURRENT_BINARY_DIR})

if(${CMAKE_VERSION} VERSION_GREATER "3.5.2")
    set(SHALLOW_CLONE "GIT_SHALLOW TRUE") # adds --depth=1 arg to git clone of External_Projects
endif()

set(THIRD_PARTY_BUILD_TYPE Release)

set(third_party_deps)

include(cmake/external/gflags.cmake)
include(cmake/external/glog.cmake)
include(cmake/external/gtest.cmake)
include(cmake/external/eigen.cmake)
include(cmake/external/cub.cmake)
include(cmake/external/boost.cmake)
include(cmake/external/protobuf.cmake)

list(APPEND third_party_deps eigen3 gflags glog gtest)

set(source_file_list)
include(cmake/common.cmake)

add_subdirectory(CUDA)
