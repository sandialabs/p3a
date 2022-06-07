#pragma once

#include "p3a_simd_common.hpp"

#include "p3a_simd_scalar.hpp"

#ifdef __AVX512F__
#include "p3a_avx512.hpp"
#endif

#include <Kokkos_Macros.hpp>

namespace p3a {

namespace simd_abi {

#if defined(__AVX512F__)
using host_native = avx512_fixed_size<8>;
#else
using host_native = scalar;
#endif

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
using device_native = scalar;
#else
using device_native = host_native;
#endif

}

template <class T>
using host_simd = simd<T, simd_abi::host_native>;
template <class T>
using device_simd = simd<T, simd_abi::device_native>;

template <class T>
using host_simd_mask = simd_mask<T, simd_abi::host_native>;
template <class T>
using device_simd_mask = simd_mask<T, simd_abi::device_native>;

}
