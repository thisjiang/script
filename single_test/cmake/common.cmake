function(cc_library TARGET_NAME)
  set(options STATIC static SHARED shared INTERFACE interface)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cc_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(WIN32)
      # add libxxx.lib prefix in windows
      set(${TARGET_NAME}_LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}${TARGET_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}" CACHE STRING "output library name for target ${TARGET_NAME}")
  endif(WIN32)
  if(cc_library_SRCS)
      if(cc_library_SHARED OR cc_library_shared) # build *.so
        add_library(${TARGET_NAME} SHARED ${cc_library_SRCS})
      elseif(cc_library_INTERFACE OR cc_library_interface)
        generate_dummy_static_lib(LIB_NAME ${TARGET_NAME} FILE_PATH ${target_SRCS} GENERATOR "generic.cmake:cc_library")
      else()
        add_library(${TARGET_NAME} STATIC ${cc_library_SRCS})
        find_fluid_modules(${TARGET_NAME})
        find_pten_modules(${TARGET_NAME})
      endif()

    if(cc_library_DEPS)
      target_link_libraries(${TARGET_NAME} ${cc_library_DEPS})
      common_link(${TARGET_NAME})
    endif()

    # cpplint code style
    foreach(source_file ${cc_library_SRCS})
      string(REGEX REPLACE "\\.[^.]*$" "" source ${source_file})
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
        list(APPEND cc_library_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
      endif()
    endforeach()

  else(cc_library_SRCS)
    if(cc_library_DEPS)
      list(REMOVE_DUPLICATES cc_library_DEPS)

      generate_dummy_static_lib(LIB_NAME ${TARGET_NAME} FILE_PATH ${target_SRCS} GENERATOR "generic.cmake:cc_library")

      target_link_libraries(${TARGET_NAME} ${cc_library_DEPS})
    else()
      message(FATAL_ERROR "Please specify source files or libraries in cc_library(${TARGET_NAME} ...).")
    endif()
  endif(cc_library_SRCS)
endfunction(cc_library)


function(cc_test TARGET_NAME)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS ARGS)
  cmake_parse_arguments(cc_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_executable(${TARGET_NAME} ${cc_test_SRCS})

  get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
  target_link_libraries(${TARGET_NAME} ${cc_test_DEPS} ${os_dependency_modules} ${third_party_deps})
  add_dependencies(${TARGET_NAME} ${cc_test_DEPS} ${third_party_deps})

  add_test(${TARGET_NAME} ${TARGET_NAME})
endfunction(cc_test)


function(cuda_library TARGET_NAME)
  set(options STATIC static SHARED shared)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cuda_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(cuda_library_SRCS)
    if (cuda_library_SHARED OR cuda_library_shared) # build *.so
      add_library(${TARGET_NAME} SHARED ${cuda_library_SRCS})
    else()
      add_library(${TARGET_NAME} STATIC ${cuda_library_SRCS})
      find_fluid_modules(${TARGET_NAME})
      find_pten_modules(${TARGET_NAME})
    endif()
    if (cuda_library_DEPS)
      add_dependencies(${TARGET_NAME} ${cuda_library_DEPS})
      target_link_libraries(${TARGET_NAME} ${cuda_library_DEPS})
    endif()
    # cpplint code style
    foreach(source_file ${cuda_library_SRCS})
      string(REGEX REPLACE "\\.[^.]*$" "" source ${source_file})
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
        list(APPEND cuda_library_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
      endif()
    endforeach()
  else(cuda_library_SRCS)
    if (cuda_library_DEPS)
      list(REMOVE_DUPLICATES cuda_library_DEPS)
      generate_dummy_static_lib(LIB_NAME ${TARGET_NAME} FILE_PATH ${target_SRCS} GENERATOR "generic.cmake:cuda_library")

      target_link_libraries(${TARGET_NAME} ${cuda_library_DEPS})
      add_dependencies(${TARGET_NAME} ${cuda_library_DEPS})
    else()
      message(FATAL "Please specify source file or library in cuda_library.")
    endif()
  endif(cuda_library_SRCS)
  if((CUDA_VERSION GREATER 9.2) AND (CUDA_VERSION LESS 11.0))
    set_target_properties(${TARGET_NAME} PROPERTIES VS_USER_PROPS ${WIN_PROPS})
  endif()
endfunction(cuda_library)

function(cuda_test TARGET_NAME)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cuda_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_executable(${TARGET_NAME} ${cuda_test_SRCS})
  get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
  target_link_libraries(${TARGET_NAME} ${cuda_test_DEPS} ${third_party_deps} ${os_dependency_modules})

  add_dependencies(${TARGET_NAME} ${cuda_test_DEPS} ${third_party_deps})

  add_test(${TARGET_NAME} ${TARGET_NAME})

  if((CUDA_VERSION GREATER 9.2) AND (CUDA_VERSION LESS 11.0))
    set_target_properties(${TARGET_NAME} PROPERTIES VS_USER_PROPS ${WIN_PROPS})
  endif()
endfunction(cuda_test)

function(add_source FILE_NAME)
  list(APPEND source_file_list ${FILE_NAME})
endfunction()
