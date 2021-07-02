#pragma once

#if defined(_MSC_VER)
#define P3A_ALWAYS_INLINE
#define P3A_NEVER_INLINE __declspec(noinline)
#else
#define P3A_ALWAYS_INLINE __attribute__((always_inline))
#define P3A_NEVER_INLINE __attribute__((noinline))
#endif

#ifdef __CUDACC__
#ifndef __CUDACC_EXTENDED_LAMBDA__
#error "P3A uses CUDA extended lambdas. Please recompile with --expt-extended-lambda"
#endif
#endif

#ifdef __HIPCC__
#ifndef __HIP_ARCH_HAS_DOUBLES__
#error "P3A requires doubles, and the HIP arch doesn't support them"
#endif
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define P3A_HOST __host__
#define P3A_DEVICE __device__
#else
#define P3A_HOST
#define P3A_DEVICE
#endif

