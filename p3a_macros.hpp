#pragma once

#include <Kokkos_Macros.hpp>

#if defined(_MSC_VER)
#define P3A_ALWAYS_INLINE
#define P3A_NEVER_INLINE __declspec(noinline)
#else
#define P3A_ALWAYS_INLINE __attribute__((always_inline))
#define P3A_NEVER_INLINE __attribute__((noinline))
#endif

#ifdef KOKKOS_ENABLE_CUDA
#ifndef __CUDACC_EXTENDED_LAMBDA__
#error "P3A uses CUDA extended lambdas. Please recompile with --expt-extended-lambda"
#endif
#endif

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#define P3A_HOST __host__
#define P3A_DEVICE __device__
#else
#define P3A_HOST
#define P3A_DEVICE
#endif

