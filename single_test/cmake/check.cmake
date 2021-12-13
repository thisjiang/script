
# check c++ 17 support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)

if(NOT COMPILER_SUPPORTS_CXX17)
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no c++17 support.")
endif()

set(CMAKE_CXX_STANDARD 17 CACHE STRING "Default C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Require C++ standard")


# check cuda version
include(CheckLanguage)
check_language(CUDA)

if (${CMAKE_CUDA_COMPILER_VERSION} LESS 11.0)
  message(FATAL_ERROR "CUDA Version should greater than 11.0, but here ${CMAKE_CUDA_COMPILER_VERSION}")
endif()

set(CMAKE_CUDA_STANDARD 14 CACHE STRING "Default CUDA standard")
set(CMAKE_CUDA_STANDARD_REQUIRED ON CACHE BOOL "Require CUDA standard")


# check python version
if(NOT PY_VERSION)
  find_package(Python3 REQUIRED)
  if(NOT Python3_FOUND)
    message(FATAL_ERROR "Python version should greate than 3.6, but cannot found.")
  endif()
  set(PY_VERSION "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")
endif()

if(${PY_VERSION} LESS 3.6)
  message(FATAL_ERROR "Python version should greate than 3.6, but here ${PY_VERSION}.")
endif()


# check git version
find_package(Git REQUIRED)
if(NOT Git_FOUND)
  message(FATAL_ERROR "Git cannot found.")
endif()

set(GIT_URL "https://github.com/")
