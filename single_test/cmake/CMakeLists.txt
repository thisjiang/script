cmake_minimum_required(VERSION 3.10)

project(third_party)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)

if(NOT COMPILER_SUPPORTS_CXX17)
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_FLAGS} has no c++17 support.")
endif()

set(THIRD_PARTY_PATH ${CMAKE_CURRENT_BINARY_DIR})
set(GIT_URL https://github.com/)

add_subdirectory(external)
