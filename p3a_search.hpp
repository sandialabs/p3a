#pragma once

#include "p3a_macros.hpp"

namespace p3a {

template <class T = void>
class less {
 public:
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
  auto operator()(T const& lhs, T const& rhs ) const
  {
    return lhs < rhs;
  }
};

template <class ForwardIt, class T, class Compare>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline
ForwardIt upper_bound(
    ForwardIt first,
    ForwardIt last,
    T const& value,
    Compare comp)
{
  auto count = last - first;
  while (count > 0) {
    auto it = first; 
    auto const step = count / 2;
    it += step;
    if (!comp(value, *it)) {
      first = ++it;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  return first;
}

template<class ForwardIt, class T, class Compare>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline
ForwardIt lower_bound(ForwardIt first, ForwardIt last, const T& value, Compare comp)
{
  auto count = last - first;
  while (count > 0) {
    auto it = first;
    auto const step = count / 2;
    it += step;
    if (comp(*it, value)) {
      first = ++it;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  return first;
}

template <class ForwardIt, class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline
ForwardIt upper_bound(
    ForwardIt first,
    ForwardIt last,
    T const& value)
{
  return upper_bound(first, last, value, p3a::less<T>());
}

template <class ForwardIt, class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline
ForwardIt lower_bound(
    ForwardIt first,
    ForwardIt last,
    T const& value)
{
  return lower_bound(first, last, value, p3a::less<T>());
}

}
