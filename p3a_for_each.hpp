#pragma once

#include <utility>

#include "p3a_execution.hpp"
#include "p3a_functions.hpp"
#include "p3a_grid3.hpp"
#include "p3a_simd.hpp"
#include "p3a_counting_iterator.hpp"

#include <Kokkos_Core.hpp>

namespace p3a {

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

template <class Integral, class UnaryFunction>
P3A_NEVER_INLINE
std::enable_if_t<std::is_integral_v<Integral>>
for_each(
    serial_execution,
    counting_iterator<Integral> first,
    counting_iterator<Integral> last,
    UnaryFunction f)
{
  for (Integral i = *first; i < *last; ++i) {
    f(i);
  }
}

template <class T, class Integral, class UnaryFunction>
P3A_NEVER_INLINE
std::enable_if_t<std::is_integral_v<Integral>>
simd_for_each(
    serial_execution,
    counting_iterator<Integral> first,
    counting_iterator<Integral> last,
    UnaryFunction f)
{
  using mask_type = host_simd_mask<T>;
  auto constexpr width = Integral(mask_type::size());
  auto const quotient = (*last - *first) / width;
  for (Integral qi = 0; qi < quotient + 1; ++qi) {
    auto const i = *first + qi * width;
    auto const lanes = minimum(width, *last - i);
    auto const mask = mask_type::first_n(lanes);
    functor(i, mask);
  }
}

template <class ForwardIt, class UnaryFunction>
P3A_NEVER_INLINE
void for_each(
    serial_execution policy,
    ForwardIt first,
    ForwardIt last,
    UnaryFunction f)
{
  for (; first != last; ++first) {
    f(*first);
  }
}

template <class T, class ForwardIt, class UnaryFunction>
P3A_NEVER_INLINE
void
simd_for_each(
    serial_execution policy,
    ForwardIt first,
    ForwardIt last,
    UnaryFunction f)
{
  auto const n = last - first;
  using Integral = std::remove_const_t<decltype(n)>;
  simd_for_each<T>(policy,
      counting_iterator<Integral>(0),
      counting_iterator<Integral>(n),
  [&] (Integral const i) P3A_ALWAYS_INLINE {
    f(first[i]);
  });
}

#ifdef __CUDACC__

template <class Integral, class UnaryFunction>
P3A_NEVER_INLINE
std::enable_if_t<std::is_integral_v<Integral>>
for_each(
    cuda_execution policy,
    counting_iterator<Integral> first,
    counting_iterator<Integral> last,
    UnaryFunction f)
{
  Integral const n = last - first;
  if (n == 0) return;
  Kokkos::parallel_for("p3a_cuda",
      Kokkos::RangePolicy<
        Kokkos::Cuda,
        Kokkos::IndexType<Integral>>(
          *first, *last),
      f);
}

template <class ForwardIt, class UnaryFunction>
P3A_NEVER_INLINE
void for_each(
    cuda_execution policy,  
    ForwardIt first,
    ForwardIt last,
    UnaryFunction f)
{
  auto const n = last - first;
  using integral_type = std::remove_const_t<decltype(n)>;
  for_each(policy,
      counting_iterator<integral_type>(0),
      counting_iterator<integral_type>(n),
  [=] __device__ (integral_type i) P3A_ALWAYS_INLINE {
    f(first[i]);
  });
}

template <class ForwardIt, class UnaryFunction>
__device__ P3A_ALWAYS_INLINE inline constexpr
void for_each(
    cuda_local_execution,
    ForwardIt const& first,
    ForwardIt const& last,
    UnaryFunction const& f)
{
  for (; first != last; ++first) {
    f(*first);
  }
}

#endif

#ifdef __HIPCC__

namespace details {

template <class F, class Integral>
__global__
void hip_for_each(F f, Integral first, Integral last) {
  auto const i = first + static_cast<Integral>(
          hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x);
  if (i < last) f(i);
}

}

template <class Integral, class UnaryFunction>
P3A_NEVER_INLINE
void for_each(
    hip_execution policy,
    counting_iterator<Integral> first,
    counting_iterator<Integral> last,
    UnaryFunction f)
{
  auto const n = last - first;
  if (n == 0) return;
  Kokkos::parallel_for("p3a_hip",
      Kokkos::RangePolicy<
        Kokkos::Experimental::HIP, 
        Kokkos::IndexType<Integral>>(
          *first, *last),
      f);
}

template <class ForwardIt, class UnaryFunction>
P3A_NEVER_INLINE
void for_each(
    hip_execution policy,
    ForwardIt first,
    ForwardIt last,
    UnaryFunction f)
{
  auto const n = last - first;
  using integral_type = std::remove_const_t<decltype(n)>;
  for_each(policy,
      counting_iterator<integral_type>(0),
      counting_iterator<integral_type>(n),
  [=] __device__ (integral_type i) P3A_ALWAYS_INLINE {
    f(first[i]);
  });
}

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

template <class Integral>
class counting_iterator3 {
 public:
  vector3<Integral> vector;
};

template <class Functor, class Integral>
P3A_ALWAYS_INLINE constexpr void for_each(
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
P3A_ALWAYS_INLINE constexpr void for_each(
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

template <class Functor, class Integral>
P3A_NEVER_INLINE void for_each(
    serial_execution,
    counting_iterator3<Integral> first,
    counting_iterator3<Integral> last,
    Functor functor)
{
  using kokkos_policy_type =
    Kokkos::MDRangePolicy<
      Kokkos::Serial,
      Kokkos::IndexType<Integral>,
      Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>;
  Kokkos::parallel_for("p3a_serial_3d",
      kokkos_policy_type(
        {first.vector.x(), first.vector.y(), first.vector.z()},
        {last.vector.x(), last.vector.y(), last.vector.z()}),
  [=] (Integral i, Integral j, Integral k) P3A_ALWAYS_INLINE {
    functor(vector3<Integral>(i, j, k));
  });
}

template <class Functor>
P3A_NEVER_INLINE void for_each(
    serial_execution policy,
    subgrid3 subgrid,
    Functor functor)
{
  for_each(policy,
      counting_iterator3<int>{subgrid.lower()},
      counting_iterator3<int>{subgrid.upper()},
      functor);
}

template <class Functor>
P3A_NEVER_INLINE void for_each(
    serial_execution policy,
    grid3 grid,
    Functor functor)
{
  for_each(policy,
      counting_iterator3<int>{vector3<int>::zero()},
      counting_iterator3<int>{grid.extents()},
      functor);
}

template <class T, class Functor, class Integral>
P3A_NEVER_INLINE void simd_for_each(
    serial_execution,
    counting_iterator3<Integral> first,
    counting_iterator3<Integral> last,
    Functor functor)
{
  using mask_type = host_simd_mask<T>;
  auto constexpr width = Integral(mask_type::size());
  auto const quotient = (last.vector.x() - first.vector.x()) / width;
  // This doesn't use Kokkos MDRangePolicy because doing so slows
  // down a user application by 10%
  for (Integral k = first.vector.z(); k < last.vector.z(); ++k) {
    for (Integral j = first.vector.y(); j < last.vector.y(); ++j) {
      for (Integral qi = 0; qi < quotient + 1; ++qi) {
        auto const i = first.vector.x() + qi * width;
        auto const lanes = minimum(width, last.vector.x() - i);
        auto const mask = mask_type::first_n(lanes);
        functor(vector3<Integral>(i, j, k), mask);
      }
    }
  }
}

template <class T, class Functor>
P3A_NEVER_INLINE void simd_for_each(
    serial_execution policy,
    subgrid3 subgrid,
    Functor functor)
{
  simd_for_each<T>(policy,
      counting_iterator3<int>{subgrid.lower()},
      counting_iterator3<int>{subgrid.upper()},
      functor);
}

template <class T, class Functor>
P3A_NEVER_INLINE void simd_for_each(
    serial_execution policy,
    grid3 grid,
    Functor functor)
{
  simd_for_each<T>(policy,
      counting_iterator3<int>{vector3<int>::zero()},
      counting_iterator3<int>{grid.extents()},
      functor);
}

#ifdef __CUDACC__

namespace details {

template <class F, class Integral>
P3A_NEVER_INLINE
void grid_for_each(
    cuda_execution policy,
    counting_iterator3<Integral> first,
    counting_iterator3<Integral> last,
    F f)
{
  auto const limits = last.vector - first.vector;
  if (limits.volume() == 0) return;
  using kokkos_policy_type =
    Kokkos::MDRangePolicy<
      Kokkos::Cuda,
      Kokkos::IndexType<Integral>,
      Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>;
  Kokkos::parallel_for("p3a_cuda_3d",
      kokkos_policy_type(
        {first.vector.x(), first.vector.y(), first.vector.z()},
        {last.vector.x(), last.vector.y(), last.vector.z()},
        {32, 1, 1}),
  [=] __device__ (Integral i, Integral j, Integral k) P3A_ALWAYS_INLINE {
    f(vector3<Integral>(i, j, k));
  });
}

template <class T, class F, class Integral>
P3A_NEVER_INLINE
void simd_grid_for_each(
    cuda_execution policy,
    counting_iterator3<Integral> first,
    counting_iterator3<Integral> last,
    F f)
{
  auto const limits = last.vector - first.vector;
  if (limits.volume() == 0) return;
  using kokkos_policy_type =
    Kokkos::MDRangePolicy<
      Kokkos::Cuda,
      Kokkos::IndexType<Integral>,
      Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>;
  Kokkos::parallel_for("p3a_cuda_3d_simd",
      kokkos_policy_type(
        {first.vector.x(), first.vector.y(), first.vector.z()},
        {last.vector.x(), last.vector.y(), last.vector.z()},
        {32, 1, 1}),
  [=] __device__ (Integral i, Integral j, Integral k) P3A_ALWAYS_INLINE {
    f(vector3<Integral>(i, j, k), device_simd_mask<T>(true));
  });
}

}

template <class F>
P3A_NEVER_INLINE
void for_each(
    cuda_execution policy,
    grid3 grid,
    F f)
{
  details::grid_for_each(policy,
      counting_iterator3<int>{vector3<int>::zero()},
      counting_iterator3<int>{grid.extents()},
      f);
}

template <class T, class F>
P3A_NEVER_INLINE
void simd_for_each(
    cuda_execution policy,
    grid3 grid,
    F f)
{
  details::simd_grid_for_each<T>(policy,
      counting_iterator3<int>{vector3<int>::zero()},
      counting_iterator3<int>{grid.extents()},
      f);
}

template <class F>
P3A_NEVER_INLINE
void for_each(
    cuda_execution policy,
    subgrid3 grid,
    F f)
{
  details::grid_for_each(policy,
      counting_iterator3<int>{grid.lower()},
      counting_iterator3<int>{grid.upper()},
      f);
}

template <class T, class F>
P3A_NEVER_INLINE
void simd_for_each(
    cuda_execution policy,
    subgrid3 grid,
    F f)
{
  details::simd_grid_for_each<T>(policy,
      counting_iterator3<int>{grid.lower()},
      counting_iterator3<int>{grid.upper()},
      f);
}

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

template <class F, class Integral>
P3A_NEVER_INLINE
void grid_for_each(
    hip_execution policy,
    counting_iterator3<Integral> first,
    counting_iterator3<Integral> last,
    F f)
{
  auto const limits = last.vector - first.vector;
  if (limits.volume() == 0) return;
  using kokkos_policy_type =
    Kokkos::MDRangePolicy<
      Kokkos::Experimental::HIP,
      Kokkos::IndexType<Integral>,
      Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>;
  Kokkos::parallel_for("p3a_hip_3d",
      kokkos_policy_type(
        {first.vector.x(), first.vector.y(), first.vector.z()},
        {last.vector.x(), last.vector.y(), last.vector.z()},
        {64, 1, 1}),
  [=] __device__ (Integral i, Integral j, Integral k) P3A_ALWAYS_INLINE {
    f(vector3<Integral>(i, j, k));
  });
}

template <class T, class F, class Integral>
P3A_NEVER_INLINE
void simd_grid_for_each(
    hip_execution policy,
    counting_iterator3<Integral> first,
    counting_iterator3<Integral> last,
    F f)
{
  auto const limits = last.vector - first.vector;
  if (limits.volume() == 0) return;
  using kokkos_policy_type =
    Kokkos::MDRangePolicy<
      Kokkos::Experimental::HIP,
      Kokkos::IndexType<Integral>,
      Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>;
  Kokkos::parallel_for("p3a_hip_3d_simd",
      kokkos_policy_type(
        {first.vector.x(), first.vector.y(), first.vector.z()},
        {last.vector.x(), last.vector.y(), last.vector.z()},
        {64, 1, 1}),
  [=] __device__ (Integral i, Integral j, Integral k) P3A_ALWAYS_INLINE {
    f(vector3<Integral>(i, j, k), device_simd_mask<T>(true));
  });
}

}

template <class F>
P3A_NEVER_INLINE
void for_each(
    hip_execution policy,
    grid3 grid,
    F f)
{
  details::grid_for_each(policy,
      counting_iterator3<int>{vector3<int>::zero()},
      counting_iterator3<int>{grid.extents()},
      f);
}

template <class T, class F>
P3A_NEVER_INLINE
void simd_for_each(
    hip_execution policy,
    grid3 grid,
    F f)
{
  details::simd_grid_for_each<T>(policy,
      counting_iterator3<int>{vector3<int>::zero()},
      counting_iterator3<int>{grid.extents()},
      f);
}

template <class F>
P3A_NEVER_INLINE
void for_each(
    hip_execution policy,
    subgrid3 grid,
    F f)
{
  details::grid_for_each(policy,
      counting_iterator3<int>{grid.lower()},
      counting_iterator3<int>{grid.upper()},
      f);
}

template <class T, class F>
P3A_NEVER_INLINE
void simd_for_each(
    hip_execution policy,
    subgrid3 grid,
    F f)
{
  details::simd_grid_for_each<T>(policy,
      counting_iterator3<int>{grid.lower()},
      counting_iterator3<int>{grid.upper()},
      f);
}

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
