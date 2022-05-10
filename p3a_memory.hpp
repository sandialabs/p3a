#pragma once

#include <iterator>

#include "p3a_macros.hpp"
#include "p3a_execution.hpp"
#include "p3a_for_each.hpp"

namespace p3a {

namespace details {

template <class InputIt, class ForwardIt>
class uninitialized_move_functor {
  InputIt m_first;
  ForwardIt m_d_first;
 public:
  using difference_type = typename std::iterator_traits<InputIt>::difference_type;
  using value_type = typename std::iterator_traits<ForwardIt>::value_type;
  uninitialized_move_functor(
      InputIt first_arg,
      ForwardIt d_first_arg)
    :m_first(first_arg)
    ,m_d_first(d_first_arg)
  {
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  void operator()(difference_type i) const
  {
    auto addr = &(m_d_first[i]);
    ::new (static_cast<void*>(addr)) value_type(std::move(m_first[i]));
  }
};

template <class InputIt, class ForwardIt>
class uninitialized_copy_functor {
  InputIt m_first;
  ForwardIt m_d_first;
 public:
  using difference_type = typename std::iterator_traits<InputIt>::difference_type;
  using value_type = typename std::iterator_traits<ForwardIt>::value_type;
  uninitialized_copy_functor(
      InputIt first_arg,
      ForwardIt d_first_arg)
    :m_first(first_arg)
    ,m_d_first(d_first_arg)
  {
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  void operator()(difference_type i) const
  {
    auto addr = &(m_d_first[i]);
    ::new (static_cast<void*>(addr)) value_type(m_first[i]);
  }
};

template <class ForwardIt>
class destroy_functor {
 public:
  using reference = typename std::iterator_traits<ForwardIt>::reference;
  using value_type = typename std::iterator_traits<ForwardIt>::value_type;
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  void operator()(reference r) const
  {
    r.~value_type();
  }
};

template <class ForwardIt>
class uninitialized_default_construct_functor {
 public:
  using reference = typename std::iterator_traits<ForwardIt>::reference;
  using value_type = typename std::iterator_traits<ForwardIt>::value_type;
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  void operator()(reference r) const
  {
    ::new (static_cast<void*>(&r)) value_type;
  }
};

template <class ForwardIt, class T>
class uninitialized_fill_functor {
  T m_value;
 public:
  uninitialized_fill_functor(T const& value_arg)
    :m_value(value_arg)
  {
  }
  using reference = typename std::iterator_traits<ForwardIt>::reference;
  using value_type = typename std::iterator_traits<ForwardIt>::value_type;
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  void operator()(reference r) const
  {
    ::new (static_cast<void*>(&r)) value_type(m_value);
  }
};

template <class InputIt, class ForwardIt>
class copy_functor {
  InputIt m_first;
  ForwardIt m_d_first;
 public:
  using difference_type = typename std::iterator_traits<InputIt>::difference_type;
  using value_type = typename std::iterator_traits<ForwardIt>::value_type;
  copy_functor(
      InputIt first_arg,
      ForwardIt d_first_arg)
    :m_first(first_arg)
    ,m_d_first(d_first_arg)
  {
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  void operator()(difference_type i) const
  {
    m_d_first[i] = m_first[i];
  }
};

template <class InputIt, class ForwardIt>
class move_functor {
  InputIt m_first;
  ForwardIt m_d_first;
 public:
  using difference_type = typename std::iterator_traits<InputIt>::difference_type;
  using value_type = typename std::iterator_traits<ForwardIt>::value_type;
  move_functor(
      InputIt first_arg,
      ForwardIt d_first_arg)
    :m_first(first_arg)
    ,m_d_first(d_first_arg)
  {
  }
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  void operator()(difference_type i) const
  {
    m_d_first[i] = std::move(m_first[i]);
  }
};

template <class ForwardIt, class T>
class fill_functor {
  T m_value;
 public:
  fill_functor(T const& value_arg)
    :m_value(value_arg)
  {
  }
  using reference = typename std::iterator_traits<ForwardIt>::reference;
  using value_type = typename std::iterator_traits<ForwardIt>::value_type;
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  void operator()(reference r) const
  {
    r = m_value;
  }
};

}

template <class ExecutionPolicy, class InputIt, class ForwardIt>
P3A_NEVER_INLINE void uninitialized_move(
    ExecutionPolicy policy,
    InputIt first,
    InputIt last,
    ForwardIt d_first)
{
  using difference_type = typename std::iterator_traits<InputIt>::difference_type;
  using functor = details::uninitialized_move_functor<InputIt, ForwardIt>;
  p3a::for_each(policy,
      counting_iterator<difference_type>(0),
      counting_iterator<difference_type>(last - first),
  functor(first, d_first));
}

template <class InputIt, class ForwardIt>
P3A_ALWAYS_INLINE inline
void uninitialized_move(
    serial_local_execution,
    InputIt first,
    InputIt const& last,
    ForwardIt d_first)
{
  for (; first != last; ++first, ++d_first) {
    ::new (static_cast<void*>(std::addressof(*d_first)))
      typename std::iterator_traits<ForwardIt>::value_type(std::move(*first));
  }
}

template <class ExecutionPolicy, class InputIt, class ForwardIt>
P3A_NEVER_INLINE void uninitialized_copy(
    ExecutionPolicy policy,
    InputIt first,
    InputIt last,
    ForwardIt d_first)
{
  using difference_type = typename std::iterator_traits<InputIt>::difference_type;
  using functor = details::uninitialized_copy_functor<InputIt, ForwardIt>;
  p3a::for_each(policy,
      counting_iterator<difference_type>(0),
      counting_iterator<difference_type>(last - first),
  functor(first, d_first));
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
P3A_NEVER_INLINE void destroy(
    ExecutionPolicy policy,
    ForwardIt first,
    ForwardIt last)
{
  using T = typename std::iterator_traits<ForwardIt>::value_type;
  if constexpr (!std::is_trivially_destructible_v<T>) {
    p3a::for_each(policy, first, last, details::destroy_functor<ForwardIt>());
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
    p3a::for_each(policy, first, last, details::uninitialized_default_construct_functor<ForwardIt>());
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
P3A_NEVER_INLINE void uninitialized_fill(
    ExecutionPolicy policy,
    ForwardIt first,
    ForwardIt last,
    T const& value)
{
  p3a::for_each(policy, first, last, details::uninitialized_fill_functor<ForwardIt, T>(value));
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

namespace details {

template <class Iterator1, class Iterator2>
bool are_in_same_space(Iterator1, Iterator2) { return true; }

template <class Iterator1, class Iterator2>
void copy_between_spaces(Iterator1, Iterator2, std::size_t)
{
  throw std::logic_error("copying between spaces with something other than pointers to the same type");
}

}

#ifdef __CUDACC__

namespace details {

template <class T, class U>
bool are_in_same_space(T* from, U* to)
{
  cudaPointerAttributes from_attributes;
  cudaPointerAttributes to_attributes;
  details::handle_cuda_error(cudaPointerGetAttributes(&from_attributes, from));
  details::handle_cuda_error(cudaPointerGetAttributes(&to_attributes, to));
  return from_attributes.type == to_attributes.type;
}

template <class T>
void copy_between_spaces(T const* from, T* to, std::size_t n)
{
  details::handle_cuda_error(
      cudaMemcpy(
        to,
        from,
        sizeof(T) * n,
        cudaMemcpyDefault));
}

}

#endif

#ifdef __HIPCC__

namespace details {

template <class T, class U>
bool are_in_same_space(T* from, U* to)
{
  hipPointerAttributes from_attributes;
  hipPointerAttributes to_attributes;
  details::handle_hip_error(hipPointerGetAttributes(&from_attributes, from));
  details::handle_hip_error(hipPointerGetAttributes(&to_attributes, to));
  return from_attributes.type == to_attributes.type;
}

template <class T>
void copy_between_spaces(T const* from, T* to, std::size_t n)
{
  details::handle_hip_error(
      hipMemcpy(
        to,
        from,
        sizeof(T) * n,
        hipMemcpyDefault));
}

}

#endif

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2>
P3A_NEVER_INLINE void copy(
    ExecutionPolicy policy,
    ForwardIt1 first,
    ForwardIt1 last,
    ForwardIt2 d_first)
{
  using value_type = typename std::iterator_traits<ForwardIt2>::value_type;
  using difference_type = typename std::iterator_traits<ForwardIt1>::difference_type;
  using functor = details::copy_functor<ForwardIt1, ForwardIt2>;
  if (details::are_in_same_space(first, d_first)) {
    p3a::for_each(policy,
        counting_iterator<difference_type>(0),
        counting_iterator<difference_type>(last - first),
        functor(first, d_first));
  } else {
    if constexpr (std::is_trivially_copyable_v<value_type>) {
      details::copy_between_spaces(first, d_first, std::size_t(last - first));
    } else {
      throw std::runtime_error("p3a::copy will not copy non-trivially-copyable objects between host and device");
    }
  }
}

template <class ForwardIt1, class ForwardIt2>
P3A_ALWAYS_INLINE inline constexpr
void copy(
    serial_local_execution,
    ForwardIt1 first,
    ForwardIt1 const& last,
    ForwardIt2 d_first)
{
  while (first != last) {
    *d_first++ = *first++;
  }
}

template <class ForwardIt1, class ForwardIt2>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
void copy(
    device_local_execution,
    ForwardIt1 first,
    ForwardIt1 const& last,
    ForwardIt2 d_first)
{
  while (first != last) {
    *d_first++ = *first++;
  }
}

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2>
P3A_NEVER_INLINE void move(
    ExecutionPolicy policy,
    ForwardIt1 first,
    ForwardIt1 last,
    ForwardIt2 d_first)
{
  using difference_type = typename std::iterator_traits<ForwardIt1>::difference_type;
  using functor = details::move_functor<ForwardIt1, ForwardIt2>;
  p3a::for_each(policy,
      counting_iterator<difference_type>(0),
      counting_iterator<difference_type>(last - first),
      functor(first, d_first));
}

template <class ForwardIt1, class ForwardIt2>
P3A_ALWAYS_INLINE inline constexpr
void move(
    serial_local_execution,
    ForwardIt1 first,
    ForwardIt1 last,
    ForwardIt2 d_first)
{
  while (first != last) {
    *d_first++ = std::move(*first++);
  }
}

template <class BidirIt1, class BidirIt2>
P3A_ALWAYS_INLINE inline constexpr
void move_backward(
    serial_local_execution,
    BidirIt1 const& first,
    BidirIt1 last,
    BidirIt2 d_last)
{
  while (first != last) {
    *(--d_last) = std::move(*(--last));
  }
}

template <class ExecutionPolicy, class ForwardIt, class T>
P3A_NEVER_INLINE void fill(
    ExecutionPolicy policy,
    ForwardIt first,
    ForwardIt last,
    T const& value)
{
  using functor = details::fill_functor<ForwardIt, T>;
  p3a::for_each(policy, first, last, functor(value));
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
    device_local_execution,
    ForwardIt first,
    ForwardIt const& last,
    T const& value)
{
  for (; first != last; ++first) {
    *first = value;
  }
}

}
