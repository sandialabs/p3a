#pragma once

#include "p3a_macros.hpp"

namespace p3a {

template <class T>
class minimizer {
 public:
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  T operator()(T const& a, T const& b) const {
    return min(a, b);
  }
};
template <class T> inline constexpr minimizer<T> minimizes = {};

template <class T>
class maximizer {
 public:
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  T operator()(T const& a, T const& b) const {
    return maximum(a, b);
  }
};
template <class T> inline constexpr maximizer<T> maximizes = {};

template <class T>
class adder {
 public:
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  T operator()(T const& a, T const& b) const {
    return a + b;
  }
};
template <class T> inline constexpr adder<T> adds = {};

template <class T>
class identity {
 public:
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr T const&
  operator()(T const& a) const {
    return a;
  }
};

}
