if (${CMAKE_CUDA_COMPILER_VERSION} GREATER_EQUAL 11.0)
  return()
endif()

include(ExternalProject)

set(CUB_PATH        "${THIRD_PARTY_PATH}/cub" CACHE STRING "A path setting for external_cub path.")
set(CUB_PREFIX_DIR  ${CUB_PATH})

set(CUB_REPOSITORY  ${GIT_URL}/NVlabs/cub.git)
set(CUB_TAG         1.8.0)

SET(CUB_INCLUDE_DIR  ${CUB_PREFIX_DIR}/src/extern_cub)
message("CUB_INCLUDE_DIR is ${CUB_INCLUDE_DIR}")
include_directories(${CUB_INCLUDE_DIR})

ExternalProject_Add(
  extern_cub
  ${EXTERNAL_PROJECT_LOG_ARGS}
  ${SHALLOW_CLONE}
  GIT_REPOSITORY  ${CUB_REPOSITORY}
  GIT_TAG         ${CUB_TAG}
  PREFIX          ${CUB_PREFIX_DIR}
  UPDATE_COMMAND    ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)

add_library(cub INTERFACE)

list(APPEND third_party_deps extern_cub)