#pragma once

#include <iterator>

#include "p3a_macros.hpp"
#include "p3a_execution.hpp"
#include "p3a_for_each.hpp"

namespace p3a {

template <class InputIt, class ForwardIt>
P3A_NEVER_INLINE void uninitialized_move(
    serial_execution,
    InputIt first,
    InputIt last,
    ForwardIt d_first)
{
  for (; first != last; ++first, ++d_first) {
    ::new (static_cast<void*>(std::addressof(*d_first)))
      typename std::iterator_traits<ForwardIt>::value_type(std::move(*first));
  }
}

#ifdef __CUDACC__

template <class InputIt, class ForwardIt>
P3A_NEVER_INLINE void uninitialized_move(
    cuda_execution policy,
    InputIt first,
    InputIt last,
    ForwardIt d_first)
{
  using value_type = typename std::iterator_traits<ForwardIt>::value_type;
  for_each(policy, first, last,
  [=] __device__ (value_type& src_value) P3A_ALWAYS_INLINE {
    auto addr = &(d_first[&src_value - &(*first)]);
    ::new (static_cast<void*>(addr)) value_type(std::move(src_value));
  });
}

#endif

#ifdef __HIPCC__

template <class InputIt, class ForwardIt>
P3A_NEVER_INLINE void uninitialized_move(
    hip_execution policy,
    InputIt first,
    InputIt last,
    ForwardIt d_first)
{
  using value_type = typename std::iterator_traits<ForwardIt>::value_type;
  for_each(policy, first, last,
  [=] __device__ (value_type& src_value) P3A_ALWAYS_INLINE {
    auto addr = &(d_first[&src_value - &(*first)]);
    ::new (static_cast<void*>(addr)) value_type(std::move(src_value));
  });
}

#endif

template <class InputIt, class ForwardIt>
P3A_NEVER_INLINE void uninitialized_copy(
    serial_execution,
    InputIt first,
    InputIt last,
    ForwardIt d_first)
{
  for (; first != last; ++first, ++d_first) {
    ::new (static_cast<void*>(std::addressof(*d_first)))
      typename std::iterator_traits<ForwardIt>::value_type(*first);
  }
}

#ifdef __CUDACC__

template <class InputIt, class ForwardIt>
P3A_NEVER_INLINE void uninitialized_copy(
    cuda_execution policy,
    InputIt first,
    InputIt last,
    ForwardIt d_first)
{
  using value_type = typename std::iterator_traits<ForwardIt>::value_type;
  for_each(policy, first, last,
  [=] P3A_DEVICE (value_type& src_value) P3A_ALWAYS_INLINE {
    auto addr = &(d_first[&src_value - &(*first)]);
    ::new (static_cast<void*>(addr)) value_type(src_value);
  });
}

#endif

template <class T>
P3A_ALWAYS_INLINE inline void destroy_at(
    serial_execution,
    T* p)
{
  p->~T();
}

template <class ForwardIt>
P3A_NEVER_INLINE void destroy(
    serial_execution policy,
    ForwardIt first,
    ForwardIt last)
{
  using T = typename std::iterator_traits<ForwardIt>::value_type;
  if constexpr (!std::is_trivially_destructible_v<T>) {
    for (; first != last; ++first) {
      destroy_at(policy, std::addressof(*first));
    }
  }
}

#ifdef __CUDACC__

template <class T>
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline
void destroy_at(cuda_execution, T* p)
{
  p->~T();
}

template <class ForwardIt>
P3A_NEVER_INLINE void destroy(
    cuda_execution policy,
    ForwardIt first,
    ForwardIt last)
{
  using T = typename std::iterator_traits<ForwardIt>::value_type;
  if constexpr (!std::is_trivially_destructible_v<T>) {
    for_each(policy, first, last,
    [=] P3A_DEVICE (T& ref) P3A_ALWAYS_INLINE {
      destroy_at(policy, &ref);
    });
  }
}

#endif

#ifdef __HIPCC__

template <class T>
__host__ __device__ P3A_ALWAYS_INLINE inline
void destroy_at(hip_execution, T* p)
{
  p->~T();
}

template <class ForwardIt>
P3A_NEVER_INLINE void destroy(
    hip_execution policy,
    ForwardIt first,
    ForwardIt last)
{
  using T = typename std::iterator_traits<ForwardIt>::value_type;
  if constexpr (!std::is_trivially_destructible_v<T>) {
    for_each(policy, first, last,
    [=] __device__ (T& ref) P3A_ALWAYS_INLINE {
      destroy_at(policy, &ref);
    });
  }
}

#endif

template <class ForwardIt>
P3A_NEVER_INLINE void uninitialized_default_construct(
    serial_execution,
    ForwardIt first,
    ForwardIt last)
{
  using T = typename std::iterator_traits<ForwardIt>::value_type;
  if constexpr (!std::is_trivially_default_constructible_v<T>) {
    for (; first != last; ++first) {
      ::new (static_cast<void*>(std::addressof(*first))) T;
    }
  }
}

#ifdef __CUDACC__

template <class ForwardIt>
P3A_NEVER_INLINE void uninitialized_default_construct(
    cuda_execution policy,
    ForwardIt first,
    ForwardIt last)
{
  using T = typename std::iterator_traits<ForwardIt>::value_type;
  if constexpr (!std::is_trivially_default_constructible_v<T>) {
    for_each(policy, first, last,
    [=] P3A_DEVICE (T& ref) P3A_ALWAYS_INLINE {
      ::new (static_cast<void*>(&ref)) T;
    });
  }
}

#endif

template <class ForwardIt, class T>
P3A_NEVER_INLINE void uninitialized_fill(
    serial_execution,
    ForwardIt first,
    ForwardIt last,
    T const& value)
{
  for (; first != last; ++first) {
    ::new (static_cast<void*>(std::addressof(*first)))
      typename std::iterator_traits<ForwardIt>::value_type(value);
  }
}

#ifdef __CUDACC__

template <class ForwardIt, class T>
P3A_NEVER_INLINE void uninitialized_fill(
    cuda_execution policy,
    ForwardIt first,
    ForwardIt last,
    T value)
{
  for_each(policy, first, last,
  [=] P3A_DEVICE (T& ref) P3A_ALWAYS_INLINE {
    ::new (static_cast<void*>(&ref)) T(value);
  });
}

#endif

template <class ForwardIt1, class ForwardIt2>
P3A_NEVER_INLINE void copy(
    serial_execution,
    ForwardIt1 first,
    ForwardIt1 last,
    ForwardIt2 d_first)
{
  while (first != last) {
    *d_first++ = *first++;
  }
}

template <class ForwardIt1, class ForwardIt2>
P3A_ALWAYS_INLINE void copy(
    serial_local_execution,
    ForwardIt1 first,
    ForwardIt1 last,
    ForwardIt2 d_first)
{
  while (first != last) {
    *d_first++ = *first++;
  }
}

#ifdef __CUDACC__

template <class ForwardIt1, class ForwardIt2>
P3A_NEVER_INLINE void copy(
    cuda_execution policy,
    ForwardIt1 first,
    ForwardIt1 last,
    ForwardIt2 d_first)
{
  using value_type = typename std::iterator_traits<ForwardIt2>::value_type;
  if constexpr (std::is_trivially_copyable_v<value_type>) {
    details::handle_cuda_error(
      cudaMemcpy(
        &*d_first,
        &*first,
        sizeof(value_type) * std::size_t(last - first), 
        cudaMemcpyDefault));
  } else {
    for_each(policy, first, last,
    [=] P3A_DEVICE (value_type& ref) P3A_ALWAYS_INLINE {
      auto& d_ref = *(d_first + (&ref - &*first));
      d_ref = ref;
    });
  }
}

template <class ForwardIt1, class ForwardIt2>
P3A_DEVICE P3A_ALWAYS_INLINE void copy(
    cuda_local_execution,
    ForwardIt1 first,
    ForwardIt1 last,
    ForwardIt2 d_first)
{
  while (first != last) {
    *d_first++ = *first++;
  }
}

#endif

#ifdef __HIPCC__

template <class ForwardIt1, class ForwardIt2>
P3A_NEVER_INLINE void copy(
    hip_execution policy,
    ForwardIt1 first,
    ForwardIt1 last,
    ForwardIt2 d_first)
{
  using value_type = typename std::iterator_traits<ForwardIt2>::value_type;
  if constexpr (std::is_trivially_copyable_v<value_type>) {
    details::handle_hip_error(
      hipMemcpy(
        &*d_first,
        &*first,
        sizeof(value_type) * std::size_t(last - first), 
        hipMemcpyDefault));
  } else {
    for_each(policy, first, last,
    [=] __device__ (value_type& ref) P3A_ALWAYS_INLINE {
      auto& d_ref = *(d_first + (&ref - &*first));
      d_ref = ref;
    });
  }
}

template <class ForwardIt1, class ForwardIt2>
__device__ P3A_ALWAYS_INLINE void copy(
    hip_local_execution,
    ForwardIt1 first,
    ForwardIt1 last,
    ForwardIt2 d_first)
{
  while (first != last) {
    *d_first++ = *first++;
  }
}

#endif

template <class ForwardIt, class T>
P3A_NEVER_INLINE void fill(
    serial_execution,
    ForwardIt first,
    ForwardIt last,
    const T& value)
{
  for (; first != last; ++first) {
    *first = value;
  }
}

template <class ForwardIt, class T>
P3A_ALWAYS_INLINE void fill(
    serial_local_execution,
    ForwardIt first,
    ForwardIt last,
    const T& value)
{
  for (; first != last; ++first) {
    *first = value;
  }
}

#ifdef __CUDACC__

template <class ForwardIt, class T>
P3A_NEVER_INLINE void fill(
    cuda_execution policy,
    ForwardIt first,
    ForwardIt last,
    T value)
{
  using value_type = typename std::iterator_traits<ForwardIt>::value_type;
  for_each(policy, first, last,
  [=] P3A_DEVICE (value_type& range_value) P3A_ALWAYS_INLINE {
    range_value = value;
  });
}

template <class ForwardIt, class T>
P3A_DEVICE P3A_ALWAYS_INLINE void fill(
    cuda_local_execution,
    ForwardIt first,
    ForwardIt last,
    const T& value)
{
  for (; first != last; ++first) {
    *first = value;
  }
}

#endif

}
