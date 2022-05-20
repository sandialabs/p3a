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

namespace details {

template <class ExecutionSpace, class Integral, class Functor>
void kokkos_for_each(
    p3a::counting_iterator<Integral> first,
    p3a::counting_iterator<Integral> last,
    Functor functor)
{
  Kokkos::parallel_for("p3a::details::kokkos_for_each(1D)",
      Kokkos::RangePolicy<
        ExecutionSpace,
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
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE auto operator()(difference_type i) const
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
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
  auto operator()(Integral i, Integral j, Integral k) const
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

template <class ForwardIt, class UnaryFunction>
P3A_ALWAYS_INLINE inline constexpr
void for_each(
    host_execution,
    ForwardIt first,
    ForwardIt const& last,
    UnaryFunction const& f)
{
  for (; first != last; ++first) {
    f(*first);
  }
}

template <class Functor, class Integral>
P3A_ALWAYS_INLINE inline constexpr void for_each(
    host_execution,
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
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline constexpr
void for_each(
    host_device_execution,
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
    host_execution policy,
    subgrid3 const& subgrid,
    Functor const& functor)
{
  for_each(policy,
      counting_iterator3<int>{subgrid.lower()},
      counting_iterator3<int>{subgrid.upper()},
      functor);
}

template <class Functor>
P3A_ALWAYS_INLINE inline constexpr void for_each(
    host_execution policy,
    grid3 const& grid,
    Functor const& functor)
{
  for_each(policy,
      counting_iterator3<int>{vector3<int>::zero()},
      counting_iterator3<int>{grid.extents()},
      functor);
}

template <class Functor>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline constexpr
void for_each(
    host_device_execution policy,
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

template <class Functor>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline constexpr
void for_each(
    host_device_execution policy,
    subgrid3 subgrid,
    Functor const& functor)
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

namespace details {

template <
  class T,
  class SimdAbi,
  class Integral,
  class OriginalFunctor>
class simd_functor {
  OriginalFunctor m_functor;
  Integral m_first_i;
  Integral m_last_i;
 public:
  simd_functor(
      OriginalFunctor functor_arg,
      Integral first_i_arg,
      Integral last_i_arg)
    :m_functor(functor_arg)
    ,m_first_i(first_i_arg)
    ,m_last_i(last_i_arg)
  {
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline auto operator()(Integral i) const
  {
    using mask_type = p3a::simd_mask<T, SimdAbi>;
    auto constexpr width = Integral(mask_type::size());
    auto const real_i = i * width + m_first_i;
    auto const lane_count = p3a::minimum(width, m_last_i - real_i);
    auto mask = mask_type(true);
    for (std::size_t i = std::size_t(lane_count); i < mask_type::size(); ++i) {
      mask[i] = false;
    }
    return m_functor(real_i, mask);
  }
};

template <
  class T,
  class SimdAbi,
  class Integral,
  class OriginalFunctor>
class simd_3d_functor {
  OriginalFunctor m_functor;
  Integral m_first_i;
  Integral m_last_i;
 public:
  simd_3d_functor(
      OriginalFunctor functor_arg,
      Integral first_i_arg,
      Integral last_i_arg)
    :m_functor(functor_arg)
    ,m_first_i(first_i_arg)
    ,m_last_i(last_i_arg)
  {
  }
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
  auto operator()(vector3<Integral> const& p) const
  {
    using mask_type = simd_mask<T, SimdAbi>;
    auto constexpr width = Integral(mask_type::size());
    auto const real_i = p.x() * width + m_first_i;
    auto const lane_count = p3a::minimum(width, m_last_i - real_i);
    auto mask = mask_type(true);
    for (std::size_t i = std::size_t(lane_count); i < mask_type::size(); ++i) {
      mask[i] = false;
    }
    return m_functor(vector3<Integral>(real_i, p.y(), p.z()), mask);
  }
};

}

template <class T, class ExecutionPolicy, class Integral, class Functor>
void simd_for_each(
    ExecutionPolicy policy,
    p3a::counting_iterator<Integral> first,
    p3a::counting_iterator<Integral> last,
    Functor functor)
{
  using simd_abi_type = typename ExecutionPolicy::simd_abi_type;
  Integral const extent = *last - *first;
  Integral constexpr width = Integral(p3a::simd<T, simd_abi_type>::size());
  Integral const quotient = extent / width;
  for_each(policy,
      counting_iterator<Integral>(0),
      counting_iterator<Integral>(quotient + 1),
  details::simd_functor<T, simd_abi_type, Integral, Functor>(functor, *first, *last));
}

template <class T, class ExecutionPolicy, class Integral, class Functor>
void simd_for_each(
    ExecutionPolicy policy,
    counting_iterator3<Integral> first,
    counting_iterator3<Integral> last,
    Functor functor)
{
  auto const extents = last.vector - first.vector;
  using simd_abi_type = typename ExecutionPolicy::simd_abi_type;
  using new_functor = details::simd_3d_functor<T, simd_abi_type, Integral, Functor>;
  Integral constexpr width = Integral(p3a::simd<T, simd_abi_type>::size());
  Integral const quotient = extents.x() / width;
  for_each(policy,
      counting_iterator3<Integral>(Integral(0), first.vector.y(), first.vector.z()),
      counting_iterator3<Integral>(Integral(quotient + 1), last.vector.y(), last.vector.z()),
      new_functor(functor, first.vector.x(), last.vector.x()));
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

}
