#pragma once

#include <Kokkos_Simd.hpp>

#include "p3a_functions.hpp"
#include "p3a_type_traits.hpp"
#include "p3a_functional.hpp"

namespace p3a {

using Kokkos::simd;
using Kokkos::simd_mask;
using Kokkos::const_where_expression;
using Kokkos::where_expression;
using Kokkos::element_aligned_tag;
namespace simd_abi = Kokkos::simd_abi;
using Kokkos::device_simd;
using Kokkos::device_simd_mask;

template <class T, class U, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
simd<T, Abi> load(T const* ptr, int i, simd_mask<U, Abi> const& mask)
{
  simd<T, Abi> result;
  where(simd_mask<T, Abi>(mask), result).copy_from(ptr + i, element_aligned_tag());
  return result;
}

template <class T, class U, class Integral, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
simd<T, Abi> load(T const* ptr, simd<Integral, Abi> const& indices, simd_mask<U, Abi> const& mask)
{
  simd<T, Abi> result;
  where(simd_mask<T, Abi>(mask), result).gather_from(ptr, indices);
  return result;
}

template <class T, class U, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
void store(simd<T, Abi> const& value, T* ptr, int i, simd_mask<U, Abi> const& mask)
{
  where(simd_mask<T, Abi>(mask), value).copy_to(ptr + i, element_aligned_tag());
}

template <class T, class U, class Integral, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
void store(simd<T, Abi> const& value, T* ptr, simd<Integral, Abi> const& indices, simd_mask<U, Abi> const& mask)
{
  where(simd_mask<T, Abi>(mask), value).scatter_to(ptr, indices);
}

template<class M, class V, class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
typename V::value_type
reduce(
    const_where_expression<M, V> const& x,
    typename V::value_type identity_element,
    maximizer<T> binary_op)
{
  return Kokkos::hmax(x);
}

template<class M, class V, class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
typename V::value_type
reduce(
    const_where_expression<M, V> const& x,
    typename V::value_type identity_element,
    adder<T> binary_op)
{
  return Kokkos::reduce(x, identity_element, std::plus<>());
}

}
