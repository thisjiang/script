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

if (WIN32)
  find_package(gflags 2.2.2 MODULE)
else()
  find_package(gflags 2.2.2)
endif()

if(GFLAGS_FOUND)
  message(STATUS "Found gflags version ${GFLAGS_VERSION}")
else()
  INCLUDE(ExternalProject)

  SET(GFLAGS_SOURCES_DIR ${THIRD_PARTY_PATH}/gflags)
  SET(GFLAGS_INSTALL_DIR ${THIRD_PARTY_PATH}/install/gflags)
  SET(GFLAGS_INCLUDE_DIR "${GFLAGS_INSTALL_DIR}/include" CACHE PATH "gflags include directory." FORCE)
  SET(GFLAGS_LIBRARIES_DIR "${GFLAGS_INSTALL_DIR}/lib/libgflags.a" CACHE FILEPATH "GFLAGS_LIBRARIES_DIR" FORCE)

  SET(GFLAGS_REPOSITORY "https://github.com/gflags/gflags.git")
  SET(GFLAGS_TAG        "v2.2.2")
  SET(GFLAGS_CONFIGURE  cd ${GFLAGS_SOURCES_DIR}/src/extern_gflags &&
                        cmake -DCMAKE_INSTALL_PREFIX=${GFLAGS_INSTALL_DIR}
                              -DBUILD_STATIC_LIBS=ON
                              -DBUILD_SHARED_LIBS=OFF
                              -DBUILD_TESTING=OFF)
  SET(GFLAGS_MAKE       cd ${GFLAGS_SOURCES_DIR}/src/extern_gflags && make)
  SET(GFLAGS_INSTALL    cd ${GFLAGS_SOURCES_DIR}/src/extern_gflags && make install)

  ExternalProject_Add(
    extern_gflags
    GIT_REPOSITORY  ${GFLAGS_REPOSITORY}
    GIT_TAG         ${GFLAGS_TAG}
    PREFIX          ${GFLAGS_SOURCES_DIR}
    CONFIGURE_COMMAND ${GFLAGS_CONFIGURE}
    BUILD_COMMAND   ${GFLAGS_MAKE}
    INSTALL_COMMAND ${GFLAGS_INSTALL}
  )

  if (WIN32)
    ADD_LIBRARY(gflags STATIC IMPORTED GLOBAL)
  endif()
  SET_PROPERTY(TARGET gflags PROPERTY IMPORTED_LOCATION ${GFLAGS_LIBRARIES_DIR})
  ADD_DEPENDENCIES(gflags extern_gflags)

  SET(GFLAGS_LIBRARIES gflags)
endif()

INCLUDE_DIRECTORIES(${GFLAGS_INCLUDE_DIR})
LIST(APPEND EXTERN_LIBRARY ${GFLAGS_LIBRARIES})
