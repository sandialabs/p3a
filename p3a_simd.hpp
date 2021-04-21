#pragma once

#include "p3a_simd_common.hpp"

#include "p3a_simd_scalar.hpp"

#ifdef __AVX512F__
#include "p3a_avx512.hpp"
#endif

namespace p3a {

namespace simd_abi {

#if defined(__AVX512F__)
using host_native = avx512;
#else
using host_native = scalar;
#endif

#if defined(__CUDACC__)
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
using host_simd_mask = typename host_simd<T>::mask_type;
template <class T>
using device_simd_mask = typename device_simd<T>::mask_type;

template <class T>
using host_simd_index = typename host_simd<T>::index_type;
template <class T>
using device_simd_index = typename device_simd<T>::index_type;

}
