#pragma once

#include "p3a_execution.hpp"
#include "p3a_functions.hpp"

namespace p3a {

template <class ForwardIt, class UnaryFunction>
CPL_ALWAYS_INLINE inline constexpr
void for_each(
    local_execution,
    ForwardIt first,
    ForwardIt last,
    UnaryFunction f)
{
  for (; first != last; ++first) {
    f(*first);
  }
}

template <class ForwardIt, class UnaryFunction>
CPL_NEVER_INLINE
void for_each(
    serial_execution,
    ForwardIt first,
    ForwardIt last,
    UnaryFunction f)
{
  for (; first != last; ++first) {
    f(*first);
  }
}

#ifdef __CUDACC__

namespace details {

template <class F, class ForwardIt>
__global__
void cuda_for_each(F f, ForwardIt first, ForwardIt last) {
  using difference_type = typename std::iterator_traits<ForwardIt>::difference_type;
  auto const i = static_cast<difference_type>(
          threadIdx.x + blockIdx.x * blockDim.x);
  ForwardIt const it = first + i;
  if (it < last) f(*it);
}

}

template <class ForwardIt, class UnaryFunction>
CPL_NEVER_INLINE
void for_each(
    cuda_execution policy,
    ForwardIt first,
    ForwardIt last,
    UnaryFunction f)
{
  auto const n = last - first;
  if (n == 0) return;
  dim3 const cuda_block(32, 1, 1);
  dim3 const cuda_grid(ceildiv(unsigned(n), cuda_block.x), 1, 1);
  std::size_t const shared_memory_bytes = 0;
  cudaStream_t const cuda_stream = policy.cuda_stream();
  details::cuda_for_each<<<
    cuda_grid,
    cuda_block,
    shared_memory_bytes,
    cuda_stream>>>(f, first, last);
}

#endif

template <class Functor>
CPL_ALWAYS_INLINE constexpr void for_each(
    local_execution,
    grid3 const& grid,
    Functor const& functor)
{
  for (int k = 0; k < grid.extents().z(); ++k) {
    for (int j = 0; j < grid.extents().y(); ++j) {
      for (int i = 0; i < grid.extents().x(); ++i) {
        functor(vector3<int>(i, j, k));
      }
    }
  }
}

template <class Functor>
CPL_NEVER_INLINE void for_each(
    serial_execution,
    grid3 grid,
    Functor functor)
{
  for (int k = 0; k < grid.extents().z(); ++k) {
    for (int j = 0; j < grid.extents().y(); ++j) {
      for (int i = 0; i < grid.extents().x(); ++i) {
        functor(vector3<int>(i, j, k));
      }
    }
  }
}

#ifdef __CUDACC__

namespace details {

template <class F>
__global__ void cuda_grid_for_each(
    F const f,
    vector3<int> const first,
    vector3<int> const last)
{
  vector3<int> index;
  index.x() = first.x() + threadIdx.x + blockIdx.x * blockDim.x;
  if (index.x() >= last.x()) return;
  index.y() = first.y() + blockIdx.y;
  index.z() = first.z() + blockIdx.z;
  f(index);
}

template <class F>
CPL_NEVER_INLINE
void grid_for_each(
    cuda_execution policy,
    vector3<int> first,
    vector3<int> last,
    F f)
{
  dim3 const cuda_block(32, 1, 1);
  auto const limits = last - first;
  if (limits.volume() == 0) return;
  dim3 const cuda_grid(
      ceildiv(unsigned(limits.x()), cuda_block.x),
      limits.y(),
      limits.z());
  std::size_t const shared_memory_bytes = 0;
  cudaStream_t const cuda_stream = policy.cuda_stream();
  details::cuda_grid_for_each<<<
    cuda_grid,
    cuda_block,
    shared_memory_bytes,
    cuda_stream>>>(f, first, last);
}

}

template <class F>
CPL_NEVER_INLINE
void for_each(
    cuda_execution policy,
    grid3 grid,
    F f)
{
  details::grid_for_each(policy, vector3<int>::zero(), grid.extents(), f);
}

#endif

template <class Functor>
CPL_NEVER_INLINE void for_each(
    serial_execution,
    subgrid3 subgrid,
    Functor functor)
{
  for (int k = subgrid.lower().z(); k < subgrid.upper().z(); ++k) {
    for (int j = subgrid.lower().y(); j < subgrid.upper().y(); ++j) {
      for (int i = subgrid.lower().x(); i < subgrid.upper().x(); ++i) {
        functor(vector3<int>(i, j, k));
      }
    }
  }
}

template <class Functor>
CPL_ALWAYS_INLINE constexpr void for_each(
    local_execution,
    subgrid3 const& subgrid,
    Functor const& functor)
{
  for (int k = subgrid.lower().z(); k < subgrid.upper().z(); ++k) {
    for (int j = subgrid.lower().y(); j < subgrid.upper().y(); ++j) {
      for (int i = subgrid.lower().x(); i < subgrid.upper().x(); ++i) {
        functor(vector3<int>(i, j, k));
      }
    }
  }
}

#ifdef __CUDACC__

template <class F>
CPL_NEVER_INLINE
void for_each(
    cuda_execution policy,
    subgrid3 grid,
    F f)
{
  details::grid_for_each(policy, grid.lower(), grid.upper(), f);
}

#endif

}
