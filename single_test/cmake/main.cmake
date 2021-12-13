# config GIT_URL with github mirrors to speed up dependent repos clone
find_package(Git REQUIRED)
option(GIT_URL "Git URL to clone dependent repos" ${GIT_URL})
if(NOT GIT_URL)
    set(GIT_URL "https://github.com/")
endif()

include(cuda.cmake)

# some necessarily macro
MACRO(UNSET_VAR VAR_NAME)
    UNSET(${VAR_NAME} CACHE)
    UNSET(${VAR_NAME})
ENDMACRO()

# import third party library
include(ExternalProject)

set(THIRD_PARTY_PATH ${CMAKE_CURRENT_BINARY_DIR})

SET(EXTERNAL_PROJECT_LOG_ARGS
    LOG_DOWNLOAD    0     # Wrap download in script to log output
    LOG_UPDATE      1     # Wrap update in script to log output
    LOG_CONFIGURE   1     # Wrap configure in script to log output
    LOG_BUILD       0     # Wrap build in script to log output
    LOG_TEST        1     # Wrap test in script to log output
    LOG_INSTALL     0     # Wrap install in script to log output
)

if(${CMAKE_VERSION} VERSION_GREATER "3.5.2")
    set(SHALLOW_CLONE "GIT_SHALLOW TRUE") # adds --depth=1 arg to git clone of External_Projects
endif()

set(THIRD_PARTY_BUILD_TYPE Release)

# add third party
include(external/gflags.cmake)
include(external/glog.cmake)
include(external/gtest.cmake)
include(external/eigen.cmake)
include(external/cub.cmake)
include(external/boost.cmake)
include(external/protobuf.cmake)

list(APPEND third_party_deps extern_eigen3 extern_gflags extern_glog extern_gtest)
