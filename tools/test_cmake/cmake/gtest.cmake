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

find_package(GTest 1.12.0 MODULE)

ENABLE_TESTING()

if(GTEST_FOUND)
  message(STATUS "Found gtest version ${GTEST_VERSION}")
else()
  INCLUDE(ExternalProject)

  SET(GTEST_SOURCES_DIR ${THIRD_PARTY_PATH}/gtest)
  SET(GTEST_INSTALL_DIR ${THIRD_PARTY_PATH}/install/gtest)
  SET(GTEST_INCLUDE_DIR "${GTEST_INSTALL_DIR}/include" CACHE PATH "gtest include directory." FORCE)
  SET(GTEST_LIBRARIES_DIR "${GTEST_INSTALL_DIR}/lib/libgtest.a" CACHE FILEPATH "gtest libraries." FORCE)
  SET(GTEST_MAIN_LIBRARIES_DIR "${GTEST_INSTALL_DIR}/lib/libgtest_main.a" CACHE FILEPATH "gtest main libraries." FORCE)

  SET(GTEST_REPOSITORY "https://github.com/google/googletest.git")
  SET(GTEST_TAG "release-1.12.0")
  SET(GTEST_CONFIGURE  cd ${GTEST_SOURCES_DIR}/src/extern_gtest &&
                        cmake -DCMAKE_INSTALL_PREFIX=${GTEST_INSTALL_DIR}
                              -DBUILD_GMOCK=ON
                              -DBUILD_SHARED_LIBS=OFF)
  SET(GTEST_MAKE       cd ${GTEST_SOURCES_DIR}/src/extern_gtest && make)
  SET(GTEST_INSTALL    cd ${GTEST_SOURCES_DIR}/src/extern_gtest && make install)

  ExternalProject_Add(
    extern_gtest
    GIT_REPOSITORY  ${GTEST_REPOSITORY}
    GIT_TAG         ${GTEST_TAG}
    PREFIX          ${GTEST_SOURCES_DIR}
    CONFIGURE_COMMAND ${GTEST_CONFIGURE}
    BUILD_COMMAND   ${GTEST_MAKE}
    INSTALL_COMMAND ${GTEST_INSTALL}
  )

  ADD_LIBRARY(gtest STATIC IMPORTED GLOBAL)
  SET_PROPERTY(TARGET gtest PROPERTY IMPORTED_LOCATION ${GTEST_LIBRARIES_DIR})
  ADD_DEPENDENCIES(gtest extern_gtest)

  ADD_LIBRARY(gtest_main STATIC IMPORTED GLOBAL)
  SET_PROPERTY(TARGET gtest_main PROPERTY IMPORTED_LOCATION ${GTEST_MAIN_LIBRARIES_DIR})
  ADD_DEPENDENCIES(gtest_main extern_gtest)

  SET(GTEST_LIBRARIES gtest)
  SET(GTEST_MAIN_LIBRARIES gtest_main)
endif()

INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIR})
LIST(APPEND TEST_LIBRARY ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
