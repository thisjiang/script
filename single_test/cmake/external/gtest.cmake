INCLUDE(GNUInstallDirs)
INCLUDE(ExternalProject)

SET(GTEST_PREFIX_DIR    ${THIRD_PARTY_PATH}/gtest)
SET(GTEST_INSTALL_DIR   ${THIRD_PARTY_PATH}/install/gtest)
SET(GTEST_INCLUDE_DIR   "${GTEST_INSTALL_DIR}/include" CACHE PATH "gtest include directory." FORCE)
set(GTEST_REPOSITORY    ${GIT_URL}/google/googletest.git)
set(GTEST_TAG           release-1.11.0)

INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIR})

IF(WIN32)
    set(GTEST_LIBRARIES
        "${GTEST_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/gtest.lib" CACHE FILEPATH "gtest libraries." FORCE)
    set(GTEST_MAIN_LIBRARIES
        "${GTEST_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/gtest_main.lib" CACHE FILEPATH "gtest main libraries." FORCE)
ELSE(WIN32)
    set(GTEST_LIBRARIES
        "${GTEST_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/libgtest.a" CACHE FILEPATH "gtest libraries." FORCE)
    set(GTEST_MAIN_LIBRARIES
        "${GTEST_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/libgtest_main.a" CACHE FILEPATH "gtest main libraries." FORCE)
ENDIF(WIN32)

ExternalProject_Add(
    extern_gtest
    ${EXTERNAL_PROJECT_LOG_ARGS}
    ${SHALLOW_CLONE}
    GIT_REPOSITORY  ${GTEST_REPOSITORY}
    GIT_TAG         ${GTEST_TAG}
    PREFIX          ${GTEST_PREFIX_DIR}
    UPDATE_COMMAND  ""
    CMAKE_ARGS      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                    -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
                    -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
                    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                    -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
                    -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
                    -DCMAKE_INSTALL_PREFIX=${GTEST_INSTALL_DIR}
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DBUILD_GMOCK=ON
                    -Dgtest_disable_pthreads=ON
                    -Dgtest_force_shared_crt=ON
                    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                    ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${GTEST_INSTALL_DIR}
                     -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                     -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
    BUILD_BYPRODUCTS ${GTEST_LIBRARIES}
    BUILD_BYPRODUCTS ${GTEST_MAIN_LIBRARIES}
)

ADD_LIBRARY(gtest STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET gtest PROPERTY IMPORTED_LOCATION ${GTEST_LIBRARIES})
ADD_DEPENDENCIES(gtest extern_gtest)

ADD_LIBRARY(gtest_main STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET gtest_main PROPERTY IMPORTED_LOCATION ${GTEST_MAIN_LIBRARIES})
ADD_DEPENDENCIES(gtest_main extern_gtest)
