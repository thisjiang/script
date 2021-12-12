include(ExternalProject)

# update eigen to the commit id f612df27 on 03/16/2021
set(EIGEN_PREFIX_DIR ${THIRD_PARTY_PATH}/eigen3)
set(EIGEN_SOURCE_DIR ${THIRD_PARTY_PATH}/eigen3/src/extern_eigen3)
set(EIGEN_REPOSITORY https://gitlab.com/libeigen/eigen.git)
set(EIGEN_TAG        3.4.0)

if(WIN32)
    add_definitions(-DEIGEN_STRONG_INLINE=inline)
endif()

set(EIGEN_INCLUDE_DIR ${EIGEN_SOURCE_DIR})
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})

ExternalProject_Add(
    extern_eigen3
    ${EXTERNAL_PROJECT_LOG_ARGS}
    ${SHALLOW_CLONE}
    GIT_REPOSITORY    ${EIGEN_REPOSITORY}
    GIT_TAG           ${EIGEN_TAG}
    PREFIX            ${EIGEN_PREFIX_DIR}
    UPDATE_COMMAND    ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
)

add_library(eigen3 INTERFACE)

add_dependencies(eigen3 extern_eigen3)
