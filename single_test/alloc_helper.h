#ifndef SCRIPT_ALLOC_HELPER_H
#define SCRIPT_ALLOC_HELPER_H

#include "memory.h"
#include "stdio.h"
#include "assert.h"
#include "string.h"

#include <type_traits>
#include <algorithm>

#include "common_func.h"
#include "cuda_helper.h"

/************************************************************************/
// CUDA Malloc
enum class Place {
    DEVICE, HOST
};

static inline bool is_host_place(const Place &p) {return p == Place::HOST;}
static inline bool is_device_place(const Place &p) {return p == Place::DEVICE;}

/************************************************************************/

class BaseAlloc {
public:
    BaseAlloc(size_t size, CUDAStream &context, Place place)
    : _size(size), _context(context), _ptr(nullptr),
      _stream(context.stream()), _place(place)
    {
    }

    BaseAlloc(const BaseAlloc &src) = delete;

    virtual ~BaseAlloc() = default;

    template<typename T>
    inline T *ptr() const {return reinterpret_cast<T*>(_ptr);}
    template<typename T>
    inline T *data() const {return reinterpret_cast<T*>(_ptr);}

    inline size_t size() const {return _size;}
    inline Place place() const {return _place;}
    inline CUDAStream& context() const {return _context;}
    inline cudaStream_t& stream() const {return _stream;}

    inline bool is_host() const {return is_host_place(this->_place);}
    inline bool is_device() const {return is_device_place(this->_place);}

    virtual void resize(size_t n, bool preserve = false) = 0;

    virtual int CopyFrom(const void *src_ptr, size_t byte_len, Place src_place) = 0;
    int CopyFromHost(const void *src_ptr, size_t byte_len) {
        return CopyFrom(src_ptr, byte_len, Place::HOST);
    }
    int CopyFromDevice(const void *src_ptr, size_t byte_len) {
        return CopyFrom(src_ptr, byte_len, Place::DEVICE);
    }

    virtual int CopyTo(void *des_ptr, size_t byte_len, Place des_place) const = 0;
    int CopyToHost(void *des_ptr, size_t byte_len) const {
        return CopyTo(des_ptr, byte_len, Place::HOST);
    }
    int CopyToDevice(void *des_ptr, size_t byte_len) const {
        return CopyTo(des_ptr, byte_len, Place::DEVICE);
    }

    virtual int CopyFrom(const BaseAlloc &src) = 0;
    virtual int CopyTo(BaseAlloc &des) const = 0;

    void SetZero() {
        if(this->is_device()) cudaMemsetAsync(_ptr, 0, _size, _stream);
        else memset(_ptr, 0, _size);
    }

    bool CheckSame(const BaseAlloc &data) const {
        assert(this->size() == data.size());
        assert(this->place() == data.place());
        bool check_res = true;
        if(this->is_host()) {
            check_res = CheckSameHost(this->_ptr, data.ptr<void>(), this->_size);
        } else {
            check_res = CheckSameDevice<char>(this->ptr<char>(), data.ptr<char>(),
                                            this->_size, this->stream());
        }
        return check_res;
    }

    template<typename T>
    T MaxError(const BaseAlloc &data) const {
        assert(this->size() == data.size());
        assert(this->place() == data.place());
        T max_err(0);
        if(this->is_host()) {
            max_err = MaxErrorHost(this->ptr<T>(), data.ptr<T>(), 
                                    this->size() / sizeof(T));
        } else {
            max_err = MaxErrorDevice(this->ptr<T>(), data.ptr<T>(), 
                                    this->size() / sizeof(T),
                                    this->stream());
        }
        return max_err;
    }

protected:
    size_t _size;
    void *_ptr;
    cudaStream_t &_stream;
    CUDAStream &_context;

    Place _place;
};

/************************************************************************/
class AllocHost : public BaseAlloc {
public:
    AllocHost(size_t size, CUDAStream &context)
    : BaseAlloc(size, context, Place::HOST)
    {
        cudaMallocHost((void**)&this->_ptr, this->_size);
    }

    virtual ~AllocHost() {
        cudaFreeHost(this->_ptr);
        _size = 0;
        _stream = 0;
        _ptr = nullptr;
    }

    int CopyFrom(const void *src_ptr, size_t byte_len, Place src_place) override {
        assert(byte_len <= this->_size);
        if(is_device_place(src_place)) {
            cudaMemcpyAsync(_ptr, src_ptr, byte_len, 
                            cudaMemcpyDeviceToHost, _stream);
        } else {
            cudaMemcpyAsync(_ptr, src_ptr, byte_len, 
                            cudaMemcpyHostToHost, _stream);
        }
        return byte_len;
    }

    int CopyTo(void *des_ptr, size_t byte_len, Place des_place) const override {
        assert(byte_len >= this->_size);
        if(is_device_place(des_place)) {
            cudaMemcpyAsync(des_ptr, _ptr, this->_size, 
                            cudaMemcpyHostToDevice, _stream);
        } else {
            cudaMemcpyAsync(des_ptr, _ptr, this->_size, 
                            cudaMemcpyHostToHost, _stream);
        }
        return _size;
    }

    int CopyFrom(const BaseAlloc &src) {
        assert(src.size() <= this->_size);
        if(is_device_place(src.place())) {
            cudaMemcpyAsync(_ptr, src.ptr<void>(), src.size(),
                            cudaMemcpyDeviceToHost, _stream);
        } else {
            cudaMemcpyAsync(_ptr, src.ptr<void>(), src.size(), 
                            cudaMemcpyHostToHost, _stream);
        }
        return src.size();
    }

    int CopyTo(BaseAlloc &des) const {
        assert(des.size() >= this->size());
        if(is_device_place(des.place())) {
            cudaMemcpyAsync(des.ptr<void>(), _ptr, _size, 
                            cudaMemcpyHostToDevice, _stream);
        } else {
            cudaMemcpyAsync(des.ptr<void>(), _ptr, _size, 
                            cudaMemcpyHostToHost, _stream);
        }
        return _size;
    }

    void resize(size_t new_size, bool preserve = false) override {
        if(new_size <= this->size()) {
            this->_size = new_size;
            return;
        }

        if(!preserve) {
            cudaFreeHost(this->_ptr);
            cudaMallocHost((void**)&this->_ptr, new_size);
        } else {
            void *tmp;
            cudaMallocHost((void**)&tmp, new_size);
            cudaMemcpyAsync(tmp, this->_ptr, this->_size, 
                            cudaMemcpyHostToHost, this->_stream);
            this->_context.sync();
            cudaFreeHost(this->_ptr);
            this->_ptr = tmp;
        }
        this->_size = new_size;
    }

    template<typename T>
    void Print(int row, int col) const override {
        ::Print<T>(this->ptr<T>(), row, col);
    }

    template<typename T>
    void Print(int num, int row, int col) const override {
        ::Print<T>(this->ptr<T>(), num, row, col);
    }

    template<typename T>
    void Print(const Dim3 &dims) const override {
        ::Print<T>(this->ptr<T>(), dims);
    }

    template<typename T, size_t D>
    void Print(const std::array<int, D> &dims) const override {
        ::Print(this->ptr<T>(), dims);
    }

    template<typename T>
    void Random() {
        ::Random<T>(this->ptr<T>(), this->_size / sizeof(T));
    }

    template<typename T>
    void Random(T a, T b) {
        ::Random<T>(this->ptr<T>(), this->_size / sizeof(T), b, a);
    }
};

/************************************************************************/
class AllocDevice: public BaseAlloc {
public:
    AllocDevice(size_t size, CUDAStream &context)
    : BaseAlloc(size, context, Place::DEVICE)
    {
        cudaMalloc((void**)&this->_ptr, this->_size);
    }

    virtual ~AllocDevice() {
        cudaFree(this->_ptr);
        _size = 0;
        _stream = 0;
        _ptr = nullptr;
    }

    int CopyFrom(const void *src_ptr, size_t byte_len, Place src_place) override {
        assert(byte_len <= this->_size);
        if(is_device_place(src_place)) {
            cudaMemcpyAsync(_ptr, src_ptr, byte_len, 
                            cudaMemcpyDeviceToDevice, _stream);
        } else {
            cudaMemcpyAsync(_ptr, src_ptr, byte_len, 
                            cudaMemcpyHostToDevice, _stream);
        }
        return byte_len;
    }

    int CopyTo(void *des_ptr, size_t byte_len, Place des_place) const override {
        assert(byte_len >= this->_size);
        if(is_device_place(des_place)) {
            cudaMemcpyAsync(des_ptr, _ptr, this->_size, 
                            cudaMemcpyDeviceToDevice, _stream);
        } else {
            cudaMemcpyAsync(des_ptr, _ptr, this->_size, 
                            cudaMemcpyDeviceToHost, _stream);
        }
        return _size;
    }

    int CopyFrom(const BaseAlloc &src) {
        assert(src.size() <= this->_size);
            if(is_device_place(src.place())) {
                cudaMemcpyAsync(_ptr, src.ptr<void>(), src.size(), 
                                cudaMemcpyDeviceToDevice, _stream);
            } else {
                cudaMemcpyAsync(_ptr, src.ptr<void>(), src.size(), 
                                cudaMemcpyHostToDevice, _stream);
            }
        return src.size();
    }

    int CopyTo(BaseAlloc &des) const {
    assert(des.size() >= this->size());
        if(is_device_place(des.place())) {
            cudaMemcpyAsync(des.ptr<void>(), _ptr, _size, 
                            cudaMemcpyDeviceToDevice, _stream);
        } else {
            cudaMemcpyAsync(des.ptr<void>(), _ptr, _size, 
                            cudaMemcpyDeviceToHost, _stream);
        }
        return _size;
    }

    void resize(size_t new_size, bool preserve = false) override {
        if(new_size <= this->size()) {
            this->_size = new_size;
            return;
        }

        if(!preserve) {
            cudaFree(this->_ptr);
            cudaMalloc((void**)&this->_ptr, new_size);
        } else {
            void *tmp;
            cudaMalloc((void**)&tmp, new_size);
            cudaMemcpyAsync(tmp, this->_ptr, this->_size, 
                            cudaMemcpyDeviceToDevice, this->_stream);
            this->_context.sync();
            cudaFree(this->_ptr);
            this->_ptr = tmp;
        }
        this->_size = new_size;
    }

    template<typename T>
    void Print(int row, int col) const override {
        AllocHost tmp(this->_size, this->_context);
        this->CopyTo(tmp);
        const char *err = this->_context.sync();
        if(err != "") {
            fprintf(stderr, "%s\n", err);
            return;
        }

        ::Print<T>(tmp.ptr<T>(), row, col);
    }

    template<typename T>
    void Print(int num, int row, int col) const override {
        AllocHost tmp(this->_size, this->_context);
        this->CopyTo(tmp);
        const char *err = this->_context.sync();
        if(err != "") {
            fprintf(stderr, "%s\n", err);
            return;
        }

        ::Print<T>(tmp.ptr<T>(), num, row, col);
    }

    template<typename T>
    void Print(const Dim3 &dims) const override {
        AllocHost tmp(this->_size, this->_context);
        this->CopyTo(tmp);
        const char *err = this->_context.sync();
        if(err != "") {
            fprintf(stderr, "%s\n", err);
            return;
        }
        ::Print<T>(tmp.ptr<T>(), dims);
    }

    template<typename T, size_t D>
    void Print(const std::array<int, D> &dims) const override {
        AllocHost tmp(this->_size, this->_context);
        this->CopyTo(tmp);
        const char *err = this->_context.sync();
        if(err != "") {
            fprintf(stderr, "%s\n", err);
            return;
        }
        ::Print(tmp.ptr<T>(), dims);
    }
};

/************************************************************************/
template<typename T>
class MallocHost : public AllocHost {
public:
    MallocHost(size_t num, CUDAStream &context)
    : AllocHost(num * sizeof(T), context)
    {
    }

    inline T *ptr() const {return reinterpret_cast<T*>(_ptr);}
    inline T *data() const {return reinterpret_cast<T*>(_ptr);}

    inline size_t num() const {return _size / sizeof(T);}

    void Print(int row, int col) const {AllocHost::Print<T>(row, col);}
    void Print(int num, int row, int col) const {AllocHost::Print<T>(num, row, col);}
    void Print(const Dim3 &dims) const {AllocHost::Print<T>(dims);}
    template<size_t D>
    void Print(const std::array<int, D> &dims) const {AllocHost::Print<T, D>(dims);}

    template<typename T2>
    int CastFrom(MallocHost<T2> &src) {
        size_t _num = this->num();
        assert(src.num() == _num);
        ConvertHost(_ptr, src.ptr(), _num);
        return _num;
    }

    template<typename T2>
    int CastTo(MallocHost<T2> &des) {
        size_t _num = num();
        assert(des.num() == _num);
        ConvertHost(des.ptr(), _ptr, _num);
        return _num;
    }

    T MaxError(const MallocHost &data) const {
        return AllocHost::MaxError<T>(data);
    }

    void Random() {
        AllocHost::Random<T>();
    }

    void Random(T a, T b) {
        AllocHost::Random<T>(a, b);
    }
};

/************************************************************************/
template<typename T>
class MallocDevice : public AllocDevice {
public:
    MallocDevice(size_t num, CUDAStream &context)
    : AllocDevice(num * sizeof(T), context)
    {
    }

    inline T *ptr() const {return reinterpret_cast<T*>(_ptr);}
    inline T *data() const {return reinterpret_cast<T*>(_ptr);}
    inline size_t num() const {return _size / sizeof(T);}

    void Print(int row, int col) const {AllocDevice::Print<T>(row, col);}
    void Print(int num, int row, int col) const {AllocDevice::Print<T>(num, row, col);}
    void Print(const Dim3 &dims) const {AllocDevice::Print<T>(dims);}
    template<size_t D>
    void Print(const std::array<int, D> &dims) const {AllocDevice::Print<T, D>(dims);}

    template<typename T2>
    int CastFrom(MallocDevice<T2> &src) {
        size_t _num = num();
        assert(src.num() == _num);
        ConvertDevice(this->ptr(), src.ptr(), _num, _stream);
        return _num;
    }

    template<typename T2>
    int CastTo(MallocDevice<T2> &des) {
        size_t _num = num();
        assert(des.num() == _num);
        ConvertDevice(des.ptr(), this->ptr(), _num, _stream);
        return _num;
    }

    T MaxError(const MallocDevice &data) const {
        return AllocDevice::MaxError<T>(data);
    }
};


#endif // SCRIPT_ALLOC_HELPER_H