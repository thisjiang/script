find_package(Boost REQUIRED COMPONENTS filesystem system)
if(Boost_FOUND)
    return()
endif()

message(WARNING
        "Boost not found, compile and install may cost a few minutes,"
        " or you can run 'apt install libboost-dev' manually.")

INCLUDE(ExternalProject)

SET(BOOST_PREFIX_DIR  ${THIRD_PARTY_PATH}/boost)
SET(BOOST_INSTALL_DIR ${THIRD_PARTY_PATH}/install/boost)
SET(BOOST_INCLUDE_DIR "${BOOST_INSTALL_DIR}/include" CACHE PATH "boost include directory." FORCE)
set(BOOST_REPOSITORY ${GIT_URL}/boostorg/boost.git)
set(BOOST_TAG "boost-1.78.0")
set(BOOST_PREFIX_DIR ${THIRD_PARTY_PATH}/boost)

if(WIN32 AND MSVC_VERSION GREATER_EQUAL 1600)
    add_definitions(-DBOOST_HAS_STATIC_ASSERT)
endif()

ExternalProject_Add(
    extern_boost
    ${EXTERNAL_PROJECT_LOG_ARGS}
    ${SHALLOW_CLONE}
    GIT_REPOSITORY  ${BOOST_REPOSITORY}
    GIT_TAG         ${BOOST_TAG}
    PREFIX          ${BOOST_PREFIX_DIR}
    UPDATE_COMMAND  ""
    BUILD_COMMAND   ${BUILD_COMMAND}
    INSTALL_COMMAND ${INSTALL_COMMAND}
    CMAKE_ARGS      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                    -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
                    -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
                    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                    -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
                    -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
                    -DBUILD_STATIC_LIBS=ON
                    -DCMAKE_INSTALL_PREFIX=${BOOST_INSTALL_DIR}
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DBUILD_TESTING=OFF
                    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                    ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${BOOST_INSTALL_DIR}
                     -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                     -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
    BUILD_BYPRODUCTS ${BOOST_LIBRARIES}
)

ADD_LIBRARY(boost STATIC IMPORTED GLOBAL)
# SET_PROPERTY(TARGET boost PROPERTY IMPORTED_LOCATION ${BOOST_LIBRARIES})
ADD_DEPENDENCIES(boost extern_boost)

list(APPEND third_party_deps extern_boost)