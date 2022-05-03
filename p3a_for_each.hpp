#pragma once

#include <utility>
#include <iterator>

#include "p3a_execution.hpp"
#include "p3a_functions.hpp"
#include "p3a_grid3.hpp"
#include "p3a_simd.hpp"
#include "p3a_counting_iterator.hpp"

#include <Kokkos_Core.hpp>

namespace p3a {

template <class Integral>
class counting_iterator3 {
 public:
  vector3<Integral> vector;
};

namespace details {

template <class ExecutionSpace, class Integral, class Functor>
void kokkos_for_each(
    p3a::counting_iterator<Integral> first,
    p3a::counting_iterator<Integral> last,
    Functor functor)
{
  Kokkos::parallel_for("p3a::details::kokkos_for_each(1D)",
      Kokkos::RangePolicy<
        ExecutionPolicy,
        Kokkos::IndexType<Integral>>(*first, *last),
      functor);
}

template <class OriginalIterator, class OriginalFunctor>
class kokkos_iterator_functor {
  OriginalIterator m_first;
  OriginalFunctor m_functor;
 public:
  using difference_type = typename std::iterator_traits<OriginalIterator>::difference_type;
  kokkos_iterator_functor(
      OriginalIterator first_arg,
      OriginalFunctor functor_arg)
    :m_first(first_arg)
    ,m_functor(functor_arg)
  {}
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE auto operator()(difference_type i) const
  {
    return m_functor(m_first[i]);
  }
};

template <class ExecutionSpace, class Iterator, class Functor>
void kokkos_for_each(
    Iterator first,
    Iterator last,
    Functor functor)
{
  using difference_type = typename std::iterator_traits<Iterator>::difference_type;
  difference_type const n = last - first;
  kokkos_for_each<ExecutionSpace>(
      p3a::counting_iterator<difference_type>(0),
      p3a::counting_iterator<difference_type>(n),
      kokkos_iterator_functor<Iterator, Functor>(first, functor));
}

template <class Integral, class OriginalFunctor>
class kokkos_3d_functor {
  OriginalFunctor m_functor;
 public:
  kokkos_3d_functor(
      OriginalFunctor functor_arg)
    :m_functor(functor_arg)
  {
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE auto operator()(Integral i, Integral j, Integral k) const
  {
    return m_functor(p3a::vector3<Integral>(i, j, k));
  }
};

template <class ExecutionSpace, class Integral, class Functor>
void kokkos_for_each(
    p3a::counting_iterator3<Integral> first,
    p3a::counting_iterator3<Integral> last,
    Functor functor)
{
  auto const limits = last.vector - first.vector;
  if (limits.volume() == 0) return;
  using kokkos_policy =
    Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::IndexType<Integral>,
      Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>;
  Kokkos::parallel_for("p3a::details::kokkos_for_each(3D)",
      kokkos_policy(
        {first.vector.x(), first.vector.y(), first.vector.z()},
        {last.vector.x(), last.vector.y(), last.vector.z()}),
  kokkos_3d_functor<Integral, Functor>(functor));
}

template <
  class T,
  class SimdAbi,
  class Integral,
  class OriginalFunctor>
class kokkos_3d_simd_functor {
  OriginalFunctor m_functor;
  Integral m_first_i;
  Integral m_last_i;
 public:
  kokkos_3d_simd_functor(
      OriginalFunctor functor_arg,
      Integral first_i_arg,
      Integral last_i_arg)
    :m_functor(functor_arg)
    ,m_first_i(first_i_arg)
    ,m_last_i(last_i_arg)
  {
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE auto operator()(Integral i, Integral j, Integral k) const
  {
    using mask_type = p3a::simd_mask<T, SimdAbi>;
    auto constexpr width = Integral(mask_type::size());
    auto const real_i = i * width + m_first_i;
    auto const lane_count = p3a::minimum(width, m_last_i - real_i);
    return m_functor(p3a::vector3<Integral>(real_i, j, k), mask_type::first_n(lane_count));
  }
};

template <class T, class SimdAbi, class ExecutionSpace, class Integral, class Functor>
void kokkos_simd_for_each(
    p3a::counting_iterator3<Integral> first,
    p3a::counting_iterator3<Integral> last,
    Functor functor)
{
  auto const extents = last.vector - first.vector;
  if (extents.volume() == 0) return;
  using new_functor = kokkos_3d_simd_functor<T, SimdAbi, Integral, Functor>;
  using kokkos_policy =
    Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::IndexType<Integral>,
      Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>;
  Integral constexpr width = Integral(p3a::simd<T, SimdAbi>::size());
  Integral const quotient = extents.x() / width;
  Kokkos::parallel_for("p3a::details::kokkos_simd_for_each(3D)",
      kokkos_policy(
        {Integral(0), first.vector.y(), first.vector.z()},
        {Integral(quotient + 1), last.vector.y(), last.vector.z()}),
  new_functor(functor, first.vector.x(), last.vector.x()));
}

template <
  class T,
  class SimdAbi,
  class Integral,
  class OriginalFunctor>
class kokkos_simd_functor {
  OriginalFunctor m_functor;
  Integral m_first_i;
  Integral m_last_i;
 public:
  kokkos_simd_functor(
      OriginalFunctor functor_arg,
      Integral first_i_arg,
      Integral last_i_arg)
    :m_functor(functor_arg)
    ,m_first_i(first_i_arg)
    ,m_last_i(last_i_arg)
  {
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE auto operator()(Integral i) const
  {
    using mask_type = p3a::simd_mask<T, SimdAbi>;
    auto constexpr width = Integral(mask_type::size());
    auto const real_i = i * width + m_first_i;
    auto const lane_count = p3a::minimum(width, m_last_i - real_i);
    return m_functor(real_i, mask_type::first_n(lane_count));
  }
};

template <class T, class SimdAbi, class ExecutionSpace, class Integral, class Functor>
void kokkos_simd_for_each(
    p3a::counting_iterator<Integral> first,
    p3a::counting_iterator<Integral> last,
    Functor functor)
{
  Integral const extent = *last - *first;
  if (extent == 0) return;
  using kokkos_policy =
    Kokkos::RangePolicy<
      ExecutionSpace,
      Kokkos::IndexType<Integral>>;
  Integral constexpr width = Integral(p3a::simd<T, SimdAbi>::size());
  Integral const quotient = extent / width;
  Kokkos::parallel_for("p3a::details::kokkos_simd_for_each(1D)",
      kokkos_policy(0, quotient + 1),
  kokkos_simd_functor<T, SimdAbi, Integral, Functor>(functor, *first, *last));
}

}

template <class ExecutionPolicy, class Iterator, class Functor>
void for_each(
    ExecutionPolicy,
    Iterator first,
    Iterator last,
    Functor functor)
{
  details::kokkos_for_each<typename ExecutionPolicy::kokkos_execution_space>(
      first, last, functor);
}

template <class T, class ExecutionPolicy, class Iterator, class Functor>
void simd_for_each(
    ExecutionPolicy,
    Iterator first,
    Iterator last,
    Functor functor)
{
  details::kokkos_simd_for_each<
    T,
    typename ExecutionPolicy::simd_abi_type,
    typename ExecutionPolicy::kokkos_execution_space>(first, last, functor);
}

template <class ForwardIt, class UnaryFunction>
P3A_ALWAYS_INLINE inline constexpr
void for_each(
    serial_local_execution,
    ForwardIt first,
    ForwardIt const& last,
    UnaryFunction const& f)
{
  for (; first != last; ++first) {
    f(*first);
  }
}

#ifdef __CUDACC__

template <class ForwardIt, class UnaryFunction>
__device__ P3A_ALWAYS_INLINE inline constexpr
void for_each(
    cuda_local_execution,
    ForwardIt first,
    ForwardIt const& last,
    UnaryFunction const& f)
{
  for (; first != last; ++first) {
    f(*first);
  }
}

#endif

#ifdef __HIPCC__

template <class ForwardIt, class UnaryFunction>
__device__ P3A_ALWAYS_INLINE inline constexpr
void for_each(
    hip_local_execution,
    ForwardIt first,
    ForwardIt const& last,
    UnaryFunction const& f)
{
  for (; first != last; ++first) {
    f(*first);
  }
}

#endif

template <class Functor, class Integral>
P3A_ALWAYS_INLINE inline constexpr void for_each(
    serial_local_execution,
    counting_iterator3<Integral> const& first,
    counting_iterator3<Integral> const& last,
    Functor const& functor)
{
  for (Integral k = first.vector.z(); k < last.vector.z(); ++k) {
    for (Integral j = first.vector.y(); j < last.vector.y(); ++j) {
      for (Integral i = first.vector.x(); i < last.vector.x(); ++i) {
        functor(vector3<Integral>(i, j, k));
      }
    }
  }
}

template <class Functor, class Integral>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
void for_each(
    local_execution,
    counting_iterator3<Integral> const& first,
    counting_iterator3<Integral> const& last,
    Functor const& functor)
{
  for (Integral k = first.vector.z(); k < last.vector.z(); ++k) {
    for (Integral j = first.vector.y(); j < last.vector.y(); ++j) {
      for (Integral i = first.vector.x(); i < last.vector.x(); ++i) {
        functor(vector3<Integral>(i, j, k));
      }
    }
  }
}

template <class Functor>
P3A_ALWAYS_INLINE inline constexpr void for_each(
    serial_local_execution policy,
    subgrid3 const& subgrid,
    Functor const& functor)
{
  for_each(policy,
      counting_iterator3<int>{subgrid.lower()},
      counting_iterator3<int>{subgrid.upper()},
      functor);
}

template <class Functor>
P3A_ALWAYS_INLINE constexpr void for_each(
    serial_local_execution policy,
    grid3 const& grid,
    Functor const& functor)
{
  for_each(policy,
      counting_iterator3<int>{vector3<int>::zero()},
      counting_iterator3<int>{grid.extents()},
      functor);
}

template <class Functor>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
void for_each(
    local_execution policy,
    grid3 const& grid,
    Functor const& functor)
{
  for_each(policy,
      counting_iterator3<int>{vector3<int>::zero()},
      counting_iterator3<int>{grid.extents()},
      functor);
}

template <class ExecutionPolicy, class Functor>
void for_each(
    ExecutionPolicy policy,
    subgrid3 subgrid,
    Functor functor)
{
  for_each(policy,
      counting_iterator3<int>{subgrid.lower()},
      counting_iterator3<int>{subgrid.upper()},
      functor);
}

template <class ExecutionPolicy, class Functor>
void for_each(
    ExecutionPolicy policy,
    grid3 grid,
    Functor functor)
{
  for_each(policy,
      counting_iterator3<int>{vector3<int>::zero()},
      counting_iterator3<int>{grid.extents()},
      functor);
}

template <class T, class ExecutionPolicy, class Functor>
P3A_NEVER_INLINE void simd_for_each(
    ExecutionPolicy policy,
    subgrid3 subgrid,
    Functor functor)
{
  simd_for_each<T>(policy,
      counting_iterator3<int>{subgrid.lower()},
      counting_iterator3<int>{subgrid.upper()},
      functor);
}

template <class T, class ExecutionPolicy, class Functor>
void simd_for_each(
    ExecutionPolicy policy,
    grid3 grid,
    Functor functor)
{
  simd_for_each<T>(policy,
      counting_iterator3<int>{vector3<int>::zero()},
      counting_iterator3<int>{grid.extents()},
      functor);
}

#ifdef __CUDACC__

template <class Functor, class Integral>
__device__ P3A_ALWAYS_INLINE constexpr void for_each(
    cuda_local_execution,
    counting_iterator3<Integral> const& first,
    counting_iterator3<Integral> const& last,
    Functor const& functor)
{
  for (Integral k = first.vector.z(); k < last.vector.z(); ++k) {
    for (Integral j = first.vector.y(); j < last.vector.y(); ++j) {
      for (Integral i = first.vector.x(); i < last.vector.x(); ++i) {
        functor(vector3<Integral>(i, j, k));
      }
    }
  }
}

template <class Functor>
__device__ P3A_ALWAYS_INLINE constexpr void for_each(
    cuda_local_execution policy,
    subgrid3 const& subgrid,
    Functor const& functor)
{
  for_each(policy,
      counting_iterator3<int>{subgrid.lower()},
      counting_iterator3<int>{subgrid.upper()},
      functor);
}

template <class Functor>
__device__ P3A_ALWAYS_INLINE constexpr void for_each(
    cuda_local_execution policy,
    grid3 const& grid,
    Functor const& functor)
{
  for_each(policy,
      counting_iterator3<int>{vector3<int>::zero()},
      counting_iterator3<int>{grid.extents()},
      functor);
}

#endif

#ifdef __HIPCC__

namespace details {

template <class Functor, class Integral>
__device__ P3A_ALWAYS_INLINE constexpr void for_each(
    hip_local_execution,
    counting_iterator3<Integral> const& first,
    counting_iterator3<Integral> const& last,
    Functor const& functor)
{
  for (Integral k = first.vector.z(); k < last.vector.z(); ++k) {
    for (Integral j = first.vector.y(); j < last.vector.y(); ++j) {
      for (Integral i = first.vector.x(); i < last.vector.x(); ++i) {
        functor(vector3<Integral>(i, j, k));
      }
    }
  }
}

template <class Functor>
__device__ P3A_ALWAYS_INLINE constexpr void for_each(
    hip_local_execution policy,
    subgrid3 const& subgrid,
    Functor const& functor)
{
  for_each(policy,
      counting_iterator3<int>{subgrid.lower()},
      counting_iterator3<int>{subgrid.upper()},
      functor);
}

template <class Functor>
__device__ P3A_ALWAYS_INLINE constexpr void for_each(
    hip_local_execution policy,
    grid3 const& grid,
    Functor const& functor)
{
  for_each(policy,
      counting_iterator3<int>{vector3<int>::zero()},
      counting_iterator3<int>{grid.extents()},
      functor);
}

#endif

}
