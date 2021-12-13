
# set cuda flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

# auto detect cuda arch
function(detect_installed_gpus out_variable)
  if(NOT CUDA_gpu_detect_output)
    set(cufile ${PROJECT_BINARY_DIR}/detect_cuda_archs.cu)

    file(WRITE ${cufile} ""
      "#include \"stdio.h\"\n"
      "#include \"cuda.h\"\n"
      "#include \"cuda_runtime.h\"\n"
      "int main() {\n"
      "  int count = 0;\n"
      "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
      "  if (count == 0) return -1;\n"
      "  for (int device = 0; device < count; ++device) {\n"
      "    cudaDeviceProp prop;\n"
      "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
      "      printf(\"%d.%d \", prop.major, prop.minor);\n"
      "  }\n"
      "  return 0;\n"
      "}\n")

    execute_process(COMMAND "${CUDA_NVCC_EXECUTABLE}"
                    "--run" "${cufile}"
                    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
                    RESULT_VARIABLE nvcc_res OUTPUT_VARIABLE nvcc_out
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(nvcc_res EQUAL 0)
      # only keep the last line of nvcc_out
      STRING(REGEX REPLACE ";" "\\\\;" nvcc_out "${nvcc_out}")
      STRING(REGEX REPLACE "\n" ";" nvcc_out "${nvcc_out}")
      list(GET nvcc_out -1 nvcc_out)
      string(REPLACE "2.1" "2.1(2.0)" nvcc_out "${nvcc_out}")
      set(CUDA_gpu_detect_output ${nvcc_out} CACHE INTERNAL "Returned GPU architetures from detect_installed_gpus tool" FORCE)
    endif()
  endif()

  if(NOT CUDA_gpu_detect_output)
    message(STATUS "Automatic GPU detection failed. Building for sm_70s architectures default.")
    set(${out_variable} 7.0 PARENT_SCOPE)
  else()
    set(${out_variable} ${CUDA_gpu_detect_output} PARENT_SCOPE)
  endif()
endfunction()

detect_installed_gpus(cuda_arch_bin)

string(REGEX REPLACE "\\." "" cuda_arch_bin "${cuda_arch_bin}")
string(REGEX MATCHALL "[0-9()]+" cuda_arch_bin "${cuda_arch_bin}")
list(REMOVE_DUPLICATES cuda_arch_bin)

set(nvcc_flags "")

# Tell NVCC to add binaries for the specified GPUs
foreach(arch ${cuda_arch_bin})
  if(arch MATCHES "([0-9]+)\\(([0-9]+)\\)")
    # User explicitly specified PTX for the concrete BIN
    string(APPEND nvcc_flags " -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1}")
  else()
    # User didn't explicitly specify PTX for the concrete BIN, we assume PTX=BIN
    string(APPEND nvcc_flags " -gencode arch=compute_${arch},code=sm_${arch}")
  endif()
endforeach()

# set nvcc flags
# set GPU architecture
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${nvcc_flags}")

# A normal CUDA stream (per thread, does not implicitly synchronize with other streams)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")
# Generate warning when an explicit stream argument is not provided in the <<<...>>> kernel launch syntax
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wdefault-stream-launch")
# Make use of fast math library.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")
# Experimental flag: Allow host code to invoke __device__constexpr functions, and device code to invoke __host__constexpr functions.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -expt-relaxed-constexpr")
# Allow __host__, __device__ annotations in lambda declarations.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -extended-lambda")
# Suppress warnings about deprecated GPU target architectures.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
# in cuda9, suppress cuda warning on eigen
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")
# Set :expt-relaxed-constexpr to suppress Eigen warnings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
# Set :expt-extended-lambda to enable HOSTDEVICE annotation on lambdas
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

message(STATUS "CUDA FLAGS: ${CMAKE_CUDA_FLAGS}")

# set build type flags
if(NOT WIN32)
  set(CMAKE_CUDA_FLAGS_DEBUG "-g")
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
  set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
  set(CMAKE_CUDA_FLAGS_MINSIZEREL "-O1 -DNDEBUG")
else()
  set(CMAKE_CUDA_FLAGS_DEBUG "-Xcompiler=\"-MDd -Zi -Ob0 -Od /RTC1\"")
  set(CMAKE_CUDA_FLAGS_RELEASE "-Xcompiler=\"-MD -O2 -Ob2\" -DNDEBUG")
  set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-Xcompiler=\"-MD -Zi -O2 -Ob1\" -DNDEBUG")
  set(CMAKE_CUDA_FLAGS_MINSIZEREL "-Xcompiler=\"-MD -O1 -Ob1\" -DNDEBUG")
endif()
