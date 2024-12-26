#pragma once

#if (defined(__CUDACC__) || defined(__HIPCC__))
#define __KERNELCC__
#endif

#if defined(__KERNELCC__)
#define INLINE inline
#define HOST __device__
#define DEVICE __device__
#define HOST_DEVICE __host__ __device__
#define KERNEL extern "C" __global__
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
#else
#define INLINE inline
#define HOST
#define DEVICE
#define HOST_DEVICE
#endif