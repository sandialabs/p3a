#pragma once

#include <Kokkos_SIMD.hpp>

#include "p3a_functions.hpp"
#include "p3a_type_traits.hpp"
#include "p3a_functional.hpp"

namespace p3a {

using Kokkos::Experimental::simd;
using Kokkos::Experimental::simd_mask;
using Kokkos::Experimental::const_where_expression;
using Kokkos::Experimental::where_expression;
using Kokkos::Experimental::element_aligned_tag;
namespace simd_abi = Kokkos::Experimental::simd_abi;
using Kokkos::Experimental::native_simd;
using Kokkos::Experimental::native_simd_mask;
using Kokkos::Experimental::condition;
using Kokkos::Experimental::where;

template <class T>
using device_simd = Kokkos::Experimental::native_simd<T>;
template <class T>
using device_simd_mask = Kokkos::Experimental::native_simd_mask<T>;

template <class T, class U, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
simd<T, Abi> load(T const* ptr, int i, simd_mask<U, Abi> const& mask)
{
  simd<T, Abi> result;
  where(simd_mask<T, Abi>(mask), result).copy_from(ptr + i, element_aligned_tag());
  return result;
}

template <class T, class U, class Integral, class Abi>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
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
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
typename V::value_type
reduce(
    const_where_expression<M, V> const& x,
    typename V::value_type identity_element,
    maximizer<T> binary_op)
{
  return Kokkos::Experimental::hmax(x);
}

template<class M, class V, class T>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
typename V::value_type
reduce(
    const_where_expression<M, V> const& x,
    typename V::value_type identity_element,
    minimizer<T> binary_op)
{
  return Kokkos::Experimental::hmin(x);
}

template<class M, class V>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
typename V::value_type reduce(
    const_where_expression<M, V> const& x,
    typename V::value_type identity_element,
    adder<typename V::value_type> binary_op)
{
  return Kokkos::Experimental::reduce(x, identity_element, std::plus<>());
}

}
