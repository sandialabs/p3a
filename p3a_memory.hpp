#pragma once

#include <iterator>

#include "p3a_macros.hpp"
#include "p3a_execution.hpp"
#include "p3a_for_each.hpp"

namespace p3a {

template <class ExecutionPolicy, class InputIt, class ForwardIt>
void uninitialized_move(
    ExecutionPolicy policy,
    InputIt first,
    InputIt last,
    ForwardIt d_first)
{
  using difference_type = typename std::iterator_traits<InputIt>::difference_type;
  using value_type = typename std::iterator_traits<ForwardIt>::value_type;
  p3a::for_each(policy,
      p3a::counting_iterator<difference_type>(0),
      p3a::counting_iterator<difference_type>(last - first),
  [=] P3A_HOST P3A_DEVICE (difference_type i) P3A_ALWAYS_INLINE {
    void* const address = static_cast<void*>(&(d_first[i]));
    ::new (address) value_type(std::move(first[i]));
  });
}

template <class InputIt, class ForwardIt>
void uninitialized_move(
    serial_execution policy,
    InputIt first,
    InputIt last,
    ForwardIt d_first)
{
  using difference_type = typename std::iterator_traits<InputIt>::difference_type;
  using value_type = typename std::iterator_traits<ForwardIt>::value_type;
  p3a::for_each(policy,
      p3a::counting_iterator<difference_type>(0),
      p3a::counting_iterator<difference_type>(last - first),
  [=] (difference_type i) P3A_ALWAYS_INLINE {
    void* const address = static_cast<void*>(std::addressof(d_first[i]));
    ::new (address) value_type(std::move(first[i]));
  });
}

template <class InputIt, class ForwardIt>
P3A_ALWAYS_INLINE inline
void uninitialized_move(
    serial_local_execution,
    InputIt first,
    InputIt const& last,
    ForwardIt d_first)
{
  using value_type = typename std::iterator_traits<ForwardIt>::value_type;
  for (; first != last; ++first, ++d_first) {
    void* const address = static_cast<void*>(std::addressof(*d_first));
    ::new (address) value_type(std::move(*first));
  }
}

template <class ExecutionPolicy, class InputIt, class ForwardIt>
void uninitialized_copy(
    ExecutionPolicy policy,
    InputIt first,
    InputIt last,
    ForwardIt d_first)
{
  using difference_type = typename std::iterator_traits<InputIt>::difference_type;
  using value_type = typename std::iterator_traits<ForwardIt>::value_type;
  p3a::for_each(policy,
      p3a::counting_iterator<difference_type>(0),
      p3a::counting_iterator<difference_type>(last - first),
  [=] P3A_HOST P3A_DEVICE (difference_type i) P3A_ALWAYS_INLINE {
    void* const address = static_cast<void*>(std::addressof(d_first[i]));
    ::new (address) value_type(first[i]);
  });
}

template <class InputIt, class ForwardIt>
void uninitialized_copy(
    serial_execution policy,
    InputIt first,
    InputIt last,
    ForwardIt d_first)
{
  using difference_type = typename std::iterator_traits<InputIt>::difference_type;
  using value_type = typename std::iterator_traits<ForwardIt>::value_type;
  p3a::for_each(policy,
      p3a::counting_iterator<difference_type>(0),
      p3a::counting_iterator<difference_type>(last - first),
  [=] (difference_type i) P3A_ALWAYS_INLINE {
    void* const address = static_cast<void*>(std::addressof(d_first[i]));
    ::new (address) value_type(first[i]);
  });
}

template <class InputIt, class ForwardIt>
P3A_ALWAYS_INLINE inline
void uninitialized_copy(
    serial_local_execution,
    InputIt first,
    InputIt const& last,
    ForwardIt d_first)
{
  for (; first != last; ++first, ++d_first) {
    ::new (static_cast<void*>(std::addressof(*d_first)))
      typename std::iterator_traits<ForwardIt>::value_type(*first);
  }
}

template <class ExecutionPolicy, class ForwardIt>
void destroy(
    ExecutionPolicy policy,
    ForwardIt first,
    ForwardIt last)
{
  using T = typename std::iterator_traits<ForwardIt>::value_type;
  if constexpr (!std::is_trivially_destructible_v<T>) {
    using difference_type = typename std::iterator_traits<ForwardIt>::difference_type;
    for_each(policy,
        counting_iterator<difference_type>(0),
        counting_iterator<difference_type>(last - first),
    [=] P3A_HOST P3A_DEVICE (difference_type i) P3A_ALWAYS_INLINE {
      first[i].~T();
    });
  }
}

template <class ForwardIt>
void destroy(
    serial_execution policy,
    ForwardIt first,
    ForwardIt last)
{
  using T = typename std::iterator_traits<ForwardIt>::value_type;
  if constexpr (!std::is_trivially_destructible_v<T>) {
    using difference_type = typename std::iterator_traits<ForwardIt>::difference_type;
    for_each(policy,
        counting_iterator<difference_type>(0),
        counting_iterator<difference_type>(last - first),
    [=] (difference_type i) P3A_ALWAYS_INLINE {
      first[i].~T();
    });
  }
}

template <class ForwardIt>
P3A_ALWAYS_INLINE inline
void destroy(
    serial_local_execution,
    ForwardIt first,
    ForwardIt const& last)
{
  using T = typename std::iterator_traits<ForwardIt>::value_type;
  if constexpr (!std::is_trivially_destructible_v<T>) {
    for (; first != last; ++first) {
      first->~T();
    }
  }
}

template <class ExecutionPolicy, class ForwardIt>
P3A_NEVER_INLINE void uninitialized_default_construct(
    ExecutionPolicy policy,
    ForwardIt first,
    ForwardIt last)
{
  using T = typename std::iterator_traits<ForwardIt>::value_type;
  if constexpr (!std::is_trivially_default_constructible_v<T>) {
    using reference = typename std::iterator_traits<ForwardIt>::reference;
    for_each(policy, first, last,
    [=] P3A_HOST P3A_DEVICE (reference r) P3A_ALWAYS_INLINE {
      ::new (static_cast<void*>(std::addressof(r))) T;
    });
  }
}

template <class ForwardIt>
P3A_ALWAYS_INLINE inline
void uninitialized_default_construct(
    serial_local_execution,
    ForwardIt first,
    ForwardIt const& last)
{
  using T = typename std::iterator_traits<ForwardIt>::value_type;
  if constexpr (!std::is_trivially_default_constructible_v<T>) {
    for (; first != last; ++first) {
      ::new (static_cast<void*>(std::addressof(*first))) T;
    }
  }
}

template <class ExecutionPolicy, class ForwardIt, class T>
void uninitialized_fill(
    ExecutionPolicy policy,
    ForwardIt first,
    ForwardIt last,
    T value)
{
  using reference = typename std::iterator_traits<ForwardIt>::reference;
  using value_type = typename std::iterator_traits<ForwardIt>::value_type;
  for_each(policy, first, last,
  [=] P3A_HOST P3A_DEVICE (reference r) P3A_ALWAYS_INLINE {
    ::new (static_cast<void*>(&r)) value_type(value);
  });
}

template <class ForwardIt, class T>
P3A_ALWAYS_INLINE inline
void uninitialized_fill(
    serial_local_execution,
    ForwardIt first,
    ForwardIt const& last,
    T const& value)
{
  for (; first != last; ++first) {
    ::new (static_cast<void*>(std::addressof(*first)))
      typename std::iterator_traits<ForwardIt>::value_type(value);
  }
}

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
P3A_ALWAYS_INLINE inline
void copy(
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
P3A_DEVICE P3A_ALWAYS_INLINE inline
void copy(
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

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2>
void move(
    ExecutionPolicy policy,
    ForwardIt1 first,
    ForwardIt1 last,
    ForwardIt2 d_first)
{
  using difference_type = typename std::iterator_traits<ForwardIt1>::difference_type;
  for_each(policy,
      counting_iterator<difference_type>(0),
      counting_iterator<difference_type>(last - first),
  [=] P3A_HOST P3A_DEVICE (difference_type i)  P3A_ALWAYS_INLINE {
    d_first[i] = std::move(first[i]);
  });
}

template <class ForwardIt1, class ForwardIt2>
void move(
    serial_execution policy,
    ForwardIt1 first,
    ForwardIt1 last,
    ForwardIt2 d_first)
{
  using difference_type = typename std::iterator_traits<ForwardIt1>::difference_type;
  for_each(policy,
      counting_iterator<difference_type>(0),
      counting_iterator<difference_type>(last - first),
  [=] (difference_type i) P3A_ALWAYS_INLINE {
    d_first[i] = std::move(first[i]);
  });
}

template <class ForwardIt1, class ForwardIt2>
P3A_ALWAYS_INLINE inline
void move(
    serial_local_execution,
    ForwardIt1 first,
    ForwardIt1 const& last,
    ForwardIt2 d_first)
{
  while (first != last) {
    *d_first++ = std::move(*first++);
  }
}

template <class BidirIt1, class BidirIt2>
void move_backward(
    serial_execution,
    BidirIt1 first,
    BidirIt1 last,
    BidirIt2 d_last)
{
  while (first != last) {
    *(--d_last) = std::move(*(--last));
  }
}

template <class BidirIt1, class BidirIt2>
P3A_ALWAYS_INLINE inline
void move_backward(
    serial_local_execution,
    BidirIt1 first,
    BidirIt1 last,
    BidirIt2 d_last)
{
  while (first != last) {
    *(--d_last) = std::move(*(--last));
  }
}


template <class ExecutionPolicy, class ForwardIt, class T>
void fill(
    ExecutionPolicy policy,
    ForwardIt first,
    ForwardIt last,
    T value)
{
  using reference = typename std::iterator_traits<ForwardIt>::reference;
  for_each(policy, first, last,
  [=] P3A_HOST P3A_DEVICE (reference r) P3A_ALWAYS_INLINE {
    r = value;
  });
}

template <class ForwardIt, class T>
void fill(
    serial_execution policy,
    ForwardIt first,
    ForwardIt last,
    T value)
{
  using reference = typename std::iterator_traits<ForwardIt>::reference;
  for_each(policy, first, last,
  [=] (reference r) P3A_ALWAYS_INLINE {
    r = value;
  });
}

template <class ForwardIt, class T>
P3A_ALWAYS_INLINE inline constexpr
void fill(
    serial_local_execution,
    ForwardIt first,
    ForwardIt const& last,
    T const& value)
{
  for (; first != last; ++first) {
    *first = value;
  }
}

template <class ForwardIt, class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
void fill(
    local_execution,
    ForwardIt first,
    ForwardIt const& last,
    T const& value)
{
  for (; first != last; ++first) {
    *first = value;
  }
}

}
