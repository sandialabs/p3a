#pragma once

#if defined(_MSC_VER)
#define P3A_ALWAYS_INLINE __forceinline
#define P3A_NEVER_INLINE __declspec(noinline)
#elif defined(__CUDACC__)
#define P3A_ALWAYS_INLINE __attribute__((always_inline))
#define P3A_NEVER_INLINE __attribute__((noinline))
#else
#define P3A_ALWAYS_INLINE __attribute__((always_inline))
#define P3A_NEVER_INLINE __attribute__((noinline))
#endif

#ifdef __CUDACC__
#define P3A_HOST __host__
#define P3A_DEVICE __device__
#else
#define P3A_HOST
#define P3A_DEVICE
#endif
