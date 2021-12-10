# Download and unpack googleflags at configure time
# See: http://crascit.com/2015/07/25/cmake-gtest/
configure_file(CMakeLists.txt.in googleflags-download/CMakeLists.txt)
# Call CMake to download and Google Log
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googleflags-download )

if(result)
  message(FATAL_ERROR "CMake step for googleflags failed: ${result}")
endif()
# Build the downloaded google log
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googleflags-download )
if(result)
  message(FATAL_ERROR "Build step for googleflags failed: ${result}")
endif()

# Add googleflags directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googleflags-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googleflags-build)
