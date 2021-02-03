#include "common_func.h"

#include "cuda_runtime.h"
#include "cuda_fp16.h"

#include <type_traits>
#include <algorithm>

#include "memory.h"
#include "stdio.h"
#include "assert.h"

// CUDAStream
class CUDAStream {
public:
    CUDAStream() {
        _stream = new cudaStream_t;
        cudaStreamCreate(_stream);

        GetDeviceAttr();
    }
    ~CUDAStream() {
        cudaStreamDestroy(*_stream);
    }

    inline cudaStream_t& stream() const {
        return *_stream;
    }

    const char* sync() {
        cudaStreamSynchronize(*_stream);
        auto err = 	cudaGetLastError();
        if(err != cudaSuccess) {
            return cudaGetErrorString(err);
        }
        return "";
    }

    inline int GetMaxThreadsPerBlock() const {return _max_block_size;}
    inline const dim3& GetCUDAMaxGridDimSize() const {return _max_grid_dim;}
    inline int GetMaxSharedSize() const {return _max_shared_size;} 
    inline int GetClockRate() const {return _clock_rate;}

private:
    inline void GetDeviceAttr() {
        cudaDeviceGetAttribute(&_max_block_size, cudaDevAttrMaxThreadsPerBlock, 0);
        cudaDeviceGetAttribute(reinterpret_cast<int *>(&_max_grid_dim.x),
                                cudaDevAttrMaxGridDimX, 0);
        cudaDeviceGetAttribute(reinterpret_cast<int *>(&_max_grid_dim.y),
                                cudaDevAttrMaxGridDimY, 0);
        cudaDeviceGetAttribute(reinterpret_cast<int *>(&_max_grid_dim.z),
                                cudaDevAttrMaxGridDimZ, 0);
        cudaDeviceGetAttribute(&_max_shared_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
        cudaDeviceGetAttribute(&_clock_rate, cudaDevAttrClockRate, 0);
    }

    cudaStream_t *_stream;
    int _max_block_size;
    dim3 _max_grid_dim;
    int _max_shared_size;
    int _clock_rate;
};

// CUDA Event
class TimeOfKernel {
private:
    TimeOfKernel() = default;
    TimeOfKernel(TimeOfKernel&) = delete;
    TimeOfKernel& operator=(const TimeOfKernel&) = delete;
    static cudaEvent_t *_start, *_stop;
    static TimeOfKernel *instance;
    static CUDAStream *_stream;

public:
    ~TimeOfKernel() {
        if(_start) {
            cudaEventDestroy(*_start);
        }
        if(_stop) {
            cudaEventDestroy(*_stop);
        }
    }

    static TimeOfKernel* get(CUDAStream &stream) {
        if(!instance) {
            instance = new TimeOfKernel;
        }
        _stream = &stream;
        return instance;
    }

    static void start() {
        if(!_start) {
            _start = new cudaEvent_t;
            cudaEventCreate(_start);
        }
        if(!_stop) {
            _stop = new cudaEvent_t;
            cudaEventCreate(_stop);
        }
        cudaEventRecord(*_start, _stream->stream());
    }

    static float stop() {
        cudaEventRecord(*_stop, _stream->stream());
        cudaEventSynchronize(*_stop);

        float time_of_kernel;
        cudaEventElapsedTime(&time_of_kernel, *_start, *_stop);
        return time_of_kernel;
    }
};

TimeOfKernel* TimeOfKernel::instance = nullptr;
cudaEvent_t* TimeOfKernel::_start = nullptr;
cudaEvent_t* TimeOfKernel::_stop = nullptr;
CUDAStream* TimeOfKernel::_stream = nullptr;


// CUDA Malloc
enum class Place {
    DEVICE, HOST
};

class BaseAlloc {
public:
    BaseAlloc(size_t size, CUDAStream &context, Place place)
    : _size(size), _context(context), _ptr(nullptr),
      _stream(context.stream()), _place(place){

      }

      BaseAlloc(const BaseAlloc&) = delete;
      virtual ~BaseAlloc() = default;

      template<typename T>
      inline T *ptr() const {return reinterpret_cast<T*>(_ptr);}
       template<typename T>
      inline T *data() const {return reinterpret_cast<T*>(_ptr);}

      inline size_t size() const {return _size;}
      virtual inline size_t num() const {return _size;}
      inline cudaStream_t& stream() const {return _stream;}

      virtual void resize(size_t n, bool preserve = false) = 0;
      virtual void print(int row, int col) = 0;

      void SetZero() {
        if(_place == Place::DEVICE) {
            cudaMemsetAsync(_ptr, 0, _size, _stream);
        } else {
            memset(_ptr, 0, _size);
        }
      }

      int CopyFrom(const void *src_ptr, size_t byte_len, Place src_place) {
        assert(byte_len <= this->_size);
        if(this->_place == Place::DEVICE) {
            if(src_place == Place::DEVICE) {
                cudaMemcpyAsync(_ptr, src_ptr, byte_len, cudaMemcpyDeviceToDevice, _stream);
            } else {
                cudaMemcpyAsync(_ptr, src_ptr, byte_len, cudaMemcpyHostToDevice, _stream);
            }
        } else {
            if(src_place == Place::DEVICE) {
                cudaMemcpyAsync(_ptr, src_ptr, byte_len, cudaMemcpyDeviceToHost, _stream);
            } else {
                cudaMemcpyAsync(_ptr, src_ptr, byte_len, cudaMemcpyHostToHost, _stream);
            }
        }
        return byte_len;
    }

    int CopyTo(void *des_ptr, size_t byte_len, Place des_place) const {
        assert(byte_len >= this->_size);
        if(this->_place == Place::DEVICE) {
            if(des_place == Place::DEVICE) {
                cudaMemcpyAsync(des_ptr, _ptr, this->_size, cudaMemcpyDeviceToDevice, _stream);
            } else {
                cudaMemcpyAsync(des_ptr, _ptr, this->_size, cudaMemcpyDeviceToHost, _stream);
            }
        } else {
            if(des_place == Place::DEVICE) {
                cudaMemcpyAsync(des_ptr, _ptr, this->_size, cudaMemcpyHostToDevice, _stream);
            } else {
                cudaMemcpyAsync(des_ptr, _ptr, this->_size, cudaMemcpyHostToHost, _stream);
            }
        }
        return _size;
    }

    int CopyFrom(BaseAlloc &src) {
        assert(src._size <= this->_size);
        if(this->_place == Place::DEVICE) {
            if(src._place == Place::DEVICE) {
                cudaMemcpyAsync(_ptr, src._ptr, src._size, cudaMemcpyDeviceToDevice, _stream);
            } else {
                cudaMemcpyAsync(_ptr, src._ptr, src._size, cudaMemcpyHostToDevice, _stream);
            }
        } else {
            if(src._place == Place::DEVICE) {
                cudaMemcpyAsync(_ptr, src._ptr, src._size, cudaMemcpyDeviceToHost, _stream);
            } else {
                cudaMemcpyAsync(_ptr, src._ptr, src._size, cudaMemcpyHostToHost, _stream);
            }
        }
        return src._size;
    }

    int CopyTo(BaseAlloc &des) {
        assert(des._size >= this->_size);
        if(this->_place == Place::DEVICE) {
            if(des._place == Place::DEVICE) {
                cudaMemcpyAsync(des._ptr, _ptr, _size, cudaMemcpyDeviceToDevice, _stream);
            } else {
                cudaMemcpyAsync(des._ptr, _ptr, _size, cudaMemcpyDeviceToHost, _stream);
            }
        } else {
            if(des._place == Place::DEVICE) {
                cudaMemcpyAsync(des._ptr, _ptr, _size, cudaMemcpyHostToDevice, _stream);
            } else {
                cudaMemcpyAsync(des._ptr, _ptr, _size, cudaMemcpyHostToHost, _stream);
            }
        }
        return _size;
    }

protected:
    size_t _size;
    void *_ptr;
    cudaStream_t &_stream;
    CUDAStream &_context;

    Place _place;
};

class AllocHost : public BaseAlloc {
public:
    AllocHost(size_t size, CUDAStream &context)
    : BaseAlloc(size, context, Place::HOST)
    {
        cudaMallocHost((void**)&this->_ptr, this->_size);
    }

    virtual ~AllocHost() {
        cudaFreeHost(this->_ptr);
    }

    void resize(size_t new_size, bool preserve = false) override {
        if(new_size < this->size()) return;

        if(!preserve) {
            cudaFreeHost(this->_ptr);
            cudaMallocHost((void**)&this->_ptr, new_size);
        } else {
            void *tmp;
            cudaMallocHost((void**)&tmp, new_size);
            cudaMemcpyAsync(tmp, this->_ptr, this->_size, cudaMemcpyHostToHost, this->_stream);
            this->_context.sync();
            cudaFreeHost(this->_ptr);
            this->_ptr = tmp;
        }
        this->_size = new_size;
    }
};

class AllocDevice: public BaseAlloc {
public:
    AllocDevice(size_t size, CUDAStream &context)
    : BaseAlloc(size, context, Place::DEVICE)
    {
        cudaMalloc((void**)&this->_ptr, this->_size);
    }

    virtual ~AllocDevice() {
        cudaFree(this->_ptr);
    }

    void resize(size_t new_size, bool preserve = false) override {
        if(new_size < this->num()) return;

        if(!preserve) {
            cudaFree(this->_ptr);
            cudaMalloc((void**)&this->_ptr, new_size);
        } else {
            void *tmp;
            cudaMalloc((void**)&tmp, new_size);
            cudaMemcpyAsync(tmp, this->_ptr, this->_size, cudaMemcpyDeviceToDevice, this->_stream);
            this->_context.sync();
            cudaFree(this->_ptr);
            this->_ptr = tmp;
        }
        this->_size = new_size;
    }
};

template<typename T>
class MallocHost : public AllocHost {
public:
    MallocHost(size_t num, CUDAStream &context)
    : AllocHost(num * sizeof(T), context)
    {
    }

    inline T *ptr() const {return reinterpret_cast<T*>(_ptr);}
    inline T *data() const {return reinterpret_cast<T*>(_ptr);}

    inline size_t num() const override {return _size / sizeof(T);}

    void print(int row, int col) override {
        printf("%d %d\n", row, col);
        for(int i = 0; i < row; i ++) {
            for(int j = 0; j < col; j ++) {
                print_ele(this->ptr()[i * col + j]);
                printf(" ");
            }
            printf("\n");
        }
    }

    template<typename T2>
    int CastFrom(MallocHost<T2> &src) {
        size_t _num = this->num();
        assert(src.num() == _num);
        KeConvertHost(_ptr, src._ptr, _num);
        return _num;
    }

    template<typename T2>
    int CastTo(MallocHost<T2> &des) {
        size_t _num = num();
        assert(des.num() == _num);
        KeConvertHost(des._ptr, _ptr, _num);
        return _num;
    }
};

template<typename T>
class MallocDevice : public AllocDevice {
public:
    MallocDevice(size_t num, CUDAStream &context)
    : AllocDevice(num * sizeof(T), context)
    {
    }

    inline T *ptr() const {return reinterpret_cast<T*>(_ptr);}
    inline T *data() const {return reinterpret_cast<T*>(_ptr);}
    inline size_t num() const override {return _size / sizeof(T);}

    void print(int row, int col) override {
        MallocHost<T> tmp(this->num(), this->_context);
        this->CopyTo(tmp);
        const char *err = this->_context.sync();
        if(err != "") {
            fprintf(stderr, "%s\n", err);
            return;
        }

        printf("%d %d\n", row, col);
        for(int i = 0; i < row; i ++) {
            for(int j = 0; j < col; j ++) {
                print_ele(tmp.ptr()[i * col + j]);
                printf(" ");
            }
            printf("\n");
        }
    }

    template<typename T2>
    int CastFrom(MallocDevice<T2> &src) {
        size_t _num = num();
        assert(src.num() == _num);
        int block = std::min(static_cast<int>(_num), 1024);
        int gird = (_num + block - 1) / block;
        KeConvertDevice<<<gird, block, 0, _stream>>>(
            _ptr, src._ptr, _num
        );
        return _num;
    }

    template<typename T2>
    int CastTo(MallocDevice<T2> &des) {
        size_t _num = num();
        assert(des.num() == _num);
        int block = std::min(static_cast<int>(_num), 1024);
        int gird = (_num + block - 1) / block;
        KeConvertDevice<<<gird, block, 0, _stream>>>(
            des._ptr, _ptr, _num
        );
        return _num;
    }
};
