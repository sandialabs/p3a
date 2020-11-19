#pragma once

#include <iterator>

#include "p3a_macros.hpp"
#include "p3a_execution.hpp"

namespace p3a {

template <class InputIt, class ForwardIt>
CPL_NEVER_INLINE void uninitialized_move(
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

template <class T>
CPL_ALWAYS_INLINE void destroy_at(
    serial_execution,
    T* p)
{
  p->~T();
}

template <class ForwardIt>
CPL_NEVER_INLINE void destroy(
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

template <class ForwardIt>
CPL_NEVER_INLINE void uninitialized_default_construct(
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

template <class ForwardIt, class T>
CPL_NEVER_INLINE void uninitialized_fill(
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

template <class ForwardIt1, class ForwardIt2>
CPL_NEVER_INLINE void copy(
    serial_execution,
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
CPL_NEVER_INLINE void copy(
    cuda_execution policy,
    ForwardIt1 first,
    ForwardIt1 last,
    ForwardIt2 d_first)
{
  using value_type = typename std::iterator_traits<ForwardIt2>::value_type;
  if constexpr (std::is_trivially_copyable_v<value_type>) {
    cudaMemcpy(
        &*d_first,
        &*first,
        sizeof(value_type) * std::size_t(last - first), 
        cudaMemcpyDefault);
  } else {
    for_each(policy, first, last,
    [=] CPL_DEVICE (value_type& ref) CPL_ALWAYS_INLINE {
      auto& d_ref = *(d_first + (&ref - &*first));
      d_ref = ref;
    });
  }
}

#endif

template <class ForwardIt, class T>
CPL_NEVER_INLINE void fill(
    serial_execution,
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
CPL_NEVER_INLINE void fill(
    cuda_execution policy,
    ForwardIt first,
    ForwardIt last,
    T value)
{
  using value_type = typename std::iterator_traits<ForwardIt>::value_type;
  for_each(policy, first, last,
  [=] CPL_DEVICE (value_type& range_value) CPL_ALWAYS_INLINE {
    range_value = value;
  });
}

#endif

}
