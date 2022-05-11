#pragma once

#include <Kokkos_Simd.hpp>

#include "p3a_functions.hpp"
#include "p3a_type_traits.hpp"

namespace p3a {

using Kokkos::simd;
using Kokkos::simd_mask;
using Kokkos::const_where_expression;
using Kokkos::where_expression;
using Kokkos::element_aligned_tag;
namespace simd_abi = Kokkos::simd_abi;

template <class T, class U, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
simd<T, Abi> load(T const* ptr, int i, simd_mask<U, Abi> const& mask)
{
  simd<T, Abi> result;
  where(simd_mask<T, Abi>(mask), result).copy_from(ptr + i, element_aligned_tag());
  return result;
}

}
