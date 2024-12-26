#pragma once

#include "types.hpp"

#if !defined(__KERNELCC__)
#include <Orochi/Orochi.h>
#endif

enum TYPED_BUFFER_TYPE
{
    TYPED_BUFFER_HOST = 0,
    TYPED_BUFFER_DEVICE = 1,
};

template <class T>
struct TypedBuffer
{
    T* m_data = nullptr;
    size_t m_size : 63;
    size_t m_isDevice : 1;

    DEVICE TypedBuffer(const TypedBuffer&) = delete;

    DEVICE void operator=(const TypedBuffer&) = delete;

#if defined(__KERNELCC__)
    DEVICE TypedBuffer() : m_size(0), m_isDevice(TYPED_BUFFER_DEVICE) {}
#else
    void allocate(size_t n)
    {
        if (m_isDevice)
        {
            if (m_data) { oroFree((oroDeviceptr)m_data); }
            oroMalloc((oroDeviceptr*)&m_data, n * sizeof(T));
        }
        else
        {
            if (m_data) { free(m_data); }
            m_data = (T*)malloc(n * sizeof(T));
        }
        m_size = n;
    }

    TypedBuffer(TYPED_BUFFER_TYPE type) : m_size(0), m_isDevice(type) {}

    ~TypedBuffer()
    {
        if (m_data)
        {
            if (m_isDevice) { oroFree((oroDeviceptr)m_data); }
            else { free(m_data); }
        }
    }

    TypedBuffer(TypedBuffer<T>&& other)
        : m_data(other.m_data),
          m_size(other.m_size),
          m_isDevice(other.m_isDevice)
    {
        other.m_data = nullptr;
        other.m_size = 0;
    }

    TypedBuffer<T> toHost() const
    {
        TypedBuffer<T> r(TYPED_BUFFER_HOST);
        r.allocate(size());
        oroMemcpyDtoH(r.data(), (oroDeviceptr)m_data, m_size * sizeof(T));
        return r;
    }
    TypedBuffer<T> toDevice() const
    {
        TypedBuffer<T> r(TYPED_BUFFER_DEVICE);
        r.allocate(size());
        oroMemcpyHtoD((oroDeviceptr)r.data(), m_data, m_size * sizeof(T));
        return r;
    }
#endif

    INLINE HOST_DEVICE size_t size() const { return m_size; }

    INLINE HOST_DEVICE size_t bytes() const { return m_size * sizeof(T); }

    INLINE HOST_DEVICE const T* data() const { return m_data; }

    INLINE HOST_DEVICE T* data() { return m_data; }

    INLINE HOST_DEVICE const T* begin() const { return data(); }

    INLINE HOST_DEVICE const T* end() const { return data() + m_size; }

    INLINE HOST_DEVICE T* begin() { return data(); }

    INLINE HOST_DEVICE T* end() { return data() + m_size; }

    INLINE HOST_DEVICE const T& operator[](int index) const
    {
        return m_data[index];
    }

    INLINE HOST_DEVICE T& operator[](int index) { return m_data[index]; }

    INLINE HOST_DEVICE bool isDevice() const { return m_isDevice; }

    INLINE HOST_DEVICE bool isHost() const { return !isDevice(); }
};