#pragma once

#include "p3a_static_array.hpp"

namespace p3a {

template <class T, int M, int N>
class static_matrix {
  static_array<static_array<T, N>, M> m_storage;
 public:
  using reference = T&;
  using const_reference = T const&;
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE static constexpr
  int row_count()
  {
    return M;
  }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE static constexpr
  int column_count()
  {
    return N;
  }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  reference operator()(int i, int j)
  {
    return m_storage[i][j];
  }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  const_reference operator()(int i, int j) const
  {
    return m_storage[i][j];
  }
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  void assign_zero()
  {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        m_storage[i][j] = T(0);
      }
    }
  }
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  void assign_identity()
  {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        m_storage[i][j] = T(i == j);
      }
    }
  }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE static constexpr
  static_matrix zero()
  {
    static_matrix result;
    result.assign_zero();
    return result;
  }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE static constexpr
  static_matrix identity()
  {
    static_matrix result;
    result.assign_identity();
    return result;
  }
};

template <class A, int M, int N, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<B>, static_matrix<decltype(A() / B()), M, N>>::type
operator/(static_matrix<A, M, N> const& a, B const& b)
{
  using result_type = decltype(a.xx() / b);
  static_matrix<result_type, M, N> result;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      result(i, j) = a(i, j) / b;
    }
  }
  return result;
}

template <class A, int M, int N, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<B>, void>::type
operator*=(static_matrix<A, M, N>& a, B const& b)
{
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      a(i, j) *= b;
    }
  }
}

}
