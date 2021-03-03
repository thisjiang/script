#ifndef SCRIPT_CUDNN_HELPER_H
#define SCRIPT_CUDNN_HELPER_H

#include "cudnn.h"

/***********************************************************/
class CUDNNHandle final {
public:
  CUDNNHandle(cudaStream_t &stream) {
    _cudnn_handle = new cudnnHandle_t;
    cudnnCreate(_cudnn_handle);
    cudnnSetStream(*_cudnn_handle, stream);
  }

  ~CUDNNHandle() {
    cudnnDestroy(*_cudnn_handle);
  }

  inline cudnnHandle_t& cudnn_handle() const {return *_cudnn_handle;}

  static inline const char* CheckError(const cudnnStatus_t &status) {
    if(status == CUDNN_STATUS_SUCCESS) return EMPTY_STRING;
    return cudnnGetErrorString(status);
  }

private:
  cudnnHandle_t *_cudnn_handle;
};
/***********************************************************/

template<typename T> inline constexpr cudnnDataType_t GetCudnnDataType(){return CUDNN_DATA_FLOAT;}
template<> inline constexpr cudnnDataType_t GetCudnnDataType<float>() {return CUDNN_DATA_FLOAT;}
template<> inline constexpr cudnnDataType_t GetCudnnDataType<half>() {return CUDNN_DATA_HALF;}
template<> inline constexpr cudnnDataType_t GetCudnnDataType<double>() {return CUDNN_DATA_DOUBLE;}

/***********************************************************/



#endif // SCRIPT_CUDNN_HELPER_H