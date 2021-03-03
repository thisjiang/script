
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand.h>


template<typename T>
curandStatus_t KeGenerateRand(curandGenerator_t &engine, T *data, size_t n) {
    static_assert(sizeof(T) != 4 || sizeof(T) != 8, "Type Not Support");
    return CURAND_STATUS_INITIALIZATION_FAILED;
}
template<>
curandStatus_t KeGenerateRand<float>(curandGenerator_t &engine, float *data, size_t n) {
    return curandGenerateUniform(engine, data, n);
}
template<>
curandStatus_t KeGenerateRand<double>(curandGenerator_t &engine, double *data, size_t n) {
    return curandGenerateUniformDouble(engine, data, n);
}

#define GENERATE_INT_RAND(T)   \
template<>  \
curandStatus_t KeGenerateRand<T>(curandGenerator_t &engine, T *data, size_t n) { \
    return curandGenerate(engine, reinterpret_cast<unsigned int*>(data), n); \
}
GENERATE_INT_RAND(int)
GENERATE_INT_RAND(unsigned int)
#undef GENERATE_INT_RAND

#define GENERATE_LONGLONG_RAND(T)   \
template<>  \
curandStatus_t KeGenerateRand<T>(curandGenerator_t &engine, T *data, size_t n) { \
    return curandGenerateLongLong(engine, reinterpret_cast<unsigned long long*>(data), n); \
}
GENERATE_LONGLONG_RAND(long long)
GENERATE_LONGLONG_RAND(unsigned long long)
#undef GENERATE_LONGLONG_RAND

template<typename T>
void RandomDevice(cudaStream_t &stream, T *data, size_t n, const T b, const T a = 0) {
    curandGenerator_t engine;
    curandGenerateSeeds(engine);
    curandSetStream(engine, stream);
    KeGenerateRand(engine, data, n);
    curandDestroyGenerator(engine);
}

/***********************************************************/