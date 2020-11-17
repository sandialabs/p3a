#pragma once

#include <iterator>

#include "p3a_execution.hpp"

namespace p3a {

template <class InputIt, class ForwardIt>
CPL_NEVER_INLINE void uninitialized_move(
    execution::serial_policy,
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
    execution::serial_policy,
    T* p)
{
  p->~T();
}

template <class ForwardIt>
CPL_NEVER_INLINE void destroy(
    execution::serial_policy policy,
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
    execution::serial_policy,
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
    execution::serial_policy,
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
    execution::serial_policy,
    ForwardIt1 first,
    ForwardIt1 last,
    ForwardIt2 d_first)
{
  while (first != last) {
    *d_first++ = *first++;
  }
}

template <class ForwardIt, class T>
CPL_NEVER_INLINE void fill(
    execution::serial_policy,
    ForwardIt first,
    ForwardIt last,
    const T& value)
{
  for (; first != last; ++first) {
    *first = value;
  }
}
}
