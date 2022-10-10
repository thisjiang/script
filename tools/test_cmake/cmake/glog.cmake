# Copyright (c) 2022 jiangcheng Authors. All Rights Reserved.
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

find_package(glog 0.6.0 MODULE)

if(GLOG_FOUND)
  message(STATUS "Found glog version ${GLOG_VERSION}")
else()
  INCLUDE(ExternalProject)

  SET(GLOG_SOURCES_DIR ${THIRD_PARTY_PATH}/glog)
  SET(GLOG_INSTALL_DIR ${THIRD_PARTY_PATH}/install/glog)
  SET(GLOG_INCLUDE_DIR "${GLOG_INSTALL_DIR}/include" CACHE PATH "glog include directory." FORCE)
  SET(GLOG_LIBRARIES_DIR "${GLOG_INSTALL_DIR}/lib/libglog.a" CACHE FILEPATH "glog library directory." FORCE)

  SET(GLOG_REPOSITORY "https://github.com/google/glog.git")
  SET(GLOG_TAG "v0.6.0")
  SET(GLOG_CONFIGURE  cd ${GLOG_SOURCES_DIR}/src/extern_glog &&
                        cmake -S . -B build -G "Unix Makefiles" &&
                        cmake -DCMAKE_INSTALL_PREFIX=${GLOG_INSTALL_DIR}
                              -DBUILD_SHARED_LIBS=OFF
                              -DWITH_GFLAGS=OFF
                              -DWITH_GTEST=OFF
                              -DBUILD_TESTING=OFF)
  SET(GLOG_MAKE       cd ${GLOG_SOURCES_DIR}/src/extern_glog && make)
  SET(GLOG_INSTALL    cd ${GLOG_SOURCES_DIR}/src/extern_glog && make install)

  ExternalProject_Add(
    extern_glog
    GIT_REPOSITORY  ${GLOG_REPOSITORY}
    GIT_TAG         ${GLOG_TAG}
    PREFIX          ${GLOG_SOURCES_DIR}
    CONFIGURE_COMMAND ${GLOG_CONFIGURE}
    BUILD_COMMAND   ${GLOG_MAKE}
    INSTALL_COMMAND ${GLOG_INSTALL}
  )

  ADD_LIBRARY(glog STATIC IMPORTED GLOBAL)
  SET_PROPERTY(TARGET glog PROPERTY IMPORTED_LOCATION ${GLOG_LIBRARIES_DIR})
  ADD_DEPENDENCIES(glog extern_glog gflags)
  LINK_LIBRARIES(glog gflags)

  SET(GLOG_LIBRARIES glog)
endif()

INCLUDE_DIRECTORIES(${GLOG_INCLUDE_DIR})
LIST(APPEND EXTERN_LIBRARY ${GLOG_LIBRARIES})
