configure_file(CMakeLists.txt.in googlelog-download/CMakeLists.txt)

execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googlelog-download )

if(result)
  message(FATAL_ERROR "CMake step for googlelog failed: ${result}")
endif()
# Build the downloaded google log
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googlelog-download )
if(result)
  message(FATAL_ERROR "Build step for googlelog failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(glog_force_shared_crt ON CACHE BOOL "" FORCE)
# Prevent installation of GTest with your project
set(INSTALL_GLOG OFF CACHE BOOL "" FORCE)

# Add googlelog directly to our build. This defines the gtest targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googlelog-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googlelog-build)

SET(GLOG_LIBRARIES "${CMAKE_CURRENT_BINARY_DIR}/googlelog-build/libglog.so" CACHE FILEPATH "glog library." FORCE)

# Add aliases for GTest and GMock libraries
if(NOT TARGET GLog::GLog)
    add_library(GLog::GLog ALIAS glog)
endif()
