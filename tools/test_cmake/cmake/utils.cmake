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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpthread -Wall -Wextra -fPIC -mavx -mfma -Wno-write-strings -Wno-psabi")

# @brief compile source code into library
# @param library name
# @param SRCS source file list
# @param DEPS dependent libraries list
function(cc_library TARGET_NAME)
  set(options STATIC static SHARED shared)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cc_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(cc_library_SRCS)
    if(cc_library_SHARED OR cc_library_shared)
      add_library(${TARGET_NAME} SHARED ${cc_library_SRCS})
    else()
      add_library(${TARGET_NAME} STATIC ${cc_library_SRCS})
    endif()

    if(cc_library_DEPS)
      target_link_libraries(${TARGET_NAME} ${cc_library_DEPS})
      add_dependencies(${TARGET_NAME} ${cc_library_DEPS})
    endif()

    # cpplint code style
    foreach(source_file ${cc_library_SRCS})
      string(REGEX REPLACE "\\.[^.]*$" "" source ${source_file})
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
        list(APPEND cc_library_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
      endif()
    endforeach()
  else()
    if(cc_library_DEPS)
      merge_static_libs(${TARGET_NAME} ${cc_library_DEPS})
    else()
      message(FATAL_ERROR "Please specify source files or libraries in cc_library(${TARGET_NAME} ...).")
    endif()
  endif(cc_library_SRCS)
endfunction(cc_library)

# @brief compile and link source code to executable file
# @param executable file name
# @param SRCS source file list
# @param DEPS dependent libraries list
# @param ARGS execute arguments list
function(cc_exec TARGET_NAME)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS ARGS)
  cmake_parse_arguments(cc_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_executable(${TARGET_NAME} ${cc_test_SRCS})
  target_link_libraries(${TARGET_NAME} ${cc_test_DEPS})
endfunction(cc_exec)

# @brief compile and link source code to google-test file
# @param executable file name
# @param SRCS source file list, whose code are writed by google test style
# @param DEPS dependent libraries
# @param ARGS execute arguments
function(cc_test TARGET_NAME)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS ARGS)
  cmake_parse_arguments(cc_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_executable(${TARGET_NAME} ${cc_test_SRCS})

  get_property(OS_MODULES GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
  target_link_libraries(${TARGET_NAME} ${OS_MODULES} ${TEST_LIBRARY} ${cc_test_DEPS})

  # No unit test should exceed 10 minutes.
  gtest_discover_tests(${TARGET_NAME}
                      EXTRA_ARGS "${cc_test_ARGS}"
                      PROPERTIES TIMEOUT 6000)
endfunction()

# @brief auto collect all header files in current directorys, no param
function(auto_collect_headers)
  file(GLOB includes LIST_DIRECTORIES false RELATIVE ${CMAKE_SOURCE_DIR} *.h)

  foreach(header ${includes})
    set(core_includes "${core_includes};${header}" CACHE INTERNAL "")
  endforeach()
endfunction()

# @brief collect source files into list
# @param source file list name
# @param SRCS source file list
function(collect_srcs SRC_GROUP)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs "SRCS")
  cmake_parse_arguments(prefix "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  foreach(cpp ${prefix_SRCS})
    set(${SRC_GROUP} "${${SRC_GROUP}};${CMAKE_CURRENT_SOURCE_DIR}/${cpp}" CACHE INTERNAL "")
  endforeach()
endfunction()

# @brief auto collect source file and generate the test excutable file
# @param DEPS dependent libraries
# @param ARGS execute arguments
# @param EXCLUDES exclude file list, no need to compile by auto_cc_test
function(auto_cc_test)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs "DEPS" "ARGS" "EXCLUDES")
  cmake_parse_arguments(AUTOTEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  file(GLOB test_files LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.c" "*.cc" "*.cpp")

  if(AUTOTEST_EXCLUDES)
    list(REMOVE_ITEM test_files ${AUTOTEST_EXCLUDES})
  endif()

  foreach(cpp ${test_files})
    get_filename_component(file_name ${cpp} NAME_WE)
    cc_test(${file_name} SRCS ${cpp} DEPS ${AUTOTEST_DEPS} ARGS ${AUTOTEST_ARGS})
  endforeach()
endfunction()
