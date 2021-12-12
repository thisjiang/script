# Download and unpack eigen at configure time
# See: http://crascit.com/2015/07/25/cmake-eigen/
configure_file(CMakeLists.txt.in eigen-download/CMakeLists.txt)

execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/eigen-download )

if(result)
  message(FATAL_ERROR "CMake step for eigen failed: ${result}")
endif()
# Build the downloaded google test
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/eigen-download )
if(result)
  message(FATAL_ERROR "Build step for eigen failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(eigen_force_shared_crt ON CACHE BOOL "" FORCE)
# Prevent installation of GTest with your project
set(INSTALL_EIGEN OFF CACHE BOOL "" FORCE)
set(INSTALL_GMOCK OFF CACHE BOOL "" FORCE)

set(EIGEN_LIBRARY "${THIRD_PARTY_PATH}/lib/libeigen.a" CACHE FILEPATH "eigen libraries." FORCE)

# Add eigen directly to our build. This defines
# the eigen and eigen_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/eigen-src
                 ${CMAKE_CURRENT_BINARY_DIR}/eigen-build)
