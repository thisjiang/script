# config GIT_URL with github mirrors to speed up dependent repos clone
find_package(Git REQUIRED)
option(GIT_URL "Git URL to clone dependent repos" ${GIT_URL})
if(NOT GIT_URL)
    set(GIT_URL "https://github.com/")
endif()

add_subdirectory(cuda)

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

# add_subdirectory(external)
include(external/glog)      # download, build, install glog
