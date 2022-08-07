#pragma once

#include "p3a_scalar.hpp"
#include "p3a_static_array.hpp"
#include "p3a_matrix3x3.hpp"

namespace p3a {

template <class T, int M, int N>
class static_matrix {
  static_array<static_array<T, N>, M> m_storage;
 public:
  using reference = T&;
  using const_reference = T const&;
  P3A_ALWAYS_INLINE static_matrix() = default;
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE static constexpr
  int row_count()
  {
    return M;
  }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE static constexpr
  int column_count()
  {
    return N;
  }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  reference operator()(int i, int j)
  {
    return m_storage[i][j];
  }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  const_reference operator()(int i, int j) const
  {
    return m_storage[i][j];
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  void assign_zero()
  {
    for (int i = 0; i < row_count(); ++i) {
      for (int j = 0; j < column_count(); ++j) {
        m_storage[i][j] = T(0);
      }
    }
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  void assign_identity()
  {
    for (int i = 0; i < row_count(); ++i) {
      for (int j = 0; j < column_count(); ++j) {
        m_storage[i][j] = T(i == j);
      }
    }
  }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE static
  static_matrix zero()
  {
    static_matrix result;
    result.assign_zero();
    return result;
  }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE static
  static_matrix identity()
  {
    static_matrix result;
    result.assign_identity();
    return result;
  }
};

template <class T>
class static_matrix<T, 3, 3> {
  static_array<static_array<T, 3>, 3> m_storage;
 public:
  using reference = T&;
  using const_reference = T const&;
  P3A_ALWAYS_INLINE static_matrix() = default;
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE static constexpr
  int row_count()
  {
    return 3;
  }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE static constexpr
  int column_count()
  {
    return 3;
  }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  reference operator()(int i, int j)
  {
    return m_storage[i][j];
  }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  const_reference operator()(int i, int j) const
  {
    return m_storage[i][j];
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  void assign_zero()
  {
    for (int i = 0; i < row_count(); ++i) {
      for (int j = 0; j < column_count(); ++j) {
        m_storage[i][j] = T(0);
      }
    }
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE constexpr
  void assign_identity()
  {
    for (int i = 0; i < row_count(); ++i) {
      for (int j = 0; j < column_count(); ++j) {
        m_storage[i][j] = T(i == j);
      }
    }
  }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE static
  static_matrix zero()
  {
    static_matrix result;
    result.assign_zero();
    return result;
  }
  [[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE static
  static_matrix identity()
  {
    static_matrix result;
    result.assign_identity();
    return result;
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline
  static_matrix(matrix3x3<T> const& a)
  {
    operator()(0, 0) = a.xx();
    operator()(0, 1) = a.xy();
    operator()(0, 2) = a.xz();
    operator()(1, 0) = a.yx();
    operator()(1, 1) = a.yy();
    operator()(1, 2) = a.yz();
    operator()(2, 0) = a.zx();
    operator()(2, 1) = a.zy();
    operator()(2, 2) = a.zz();
  }
  P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline explicit constexpr
  operator matrix3x3<T> () const
  {
    return matrix3x3<T>(
      operator()(0, 0),
      operator()(0, 1),
      operator()(0, 2),
      operator()(1, 0),
      operator()(1, 1),
      operator()(1, 2),
      operator()(2, 0),
      operator()(2, 1),
      operator()(2, 2));
  }
};

template <class A, int M, int N, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline
typename std::enable_if<is_scalar<B>, static_matrix<decltype(A() / B()), M, N>>::type
operator/(static_matrix<A, M, N> const& a, B const& b)
{
  using result_type = decltype(a(0, 0) / b);
  static_matrix<result_type, M, N> result;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      result(i, j) = a(i, j) / b;
    }
  }
  return result;
}

template <class A, int M, int N, class B>
[[nodiscard]] P3A_HOST_DEVICE P3A_ALWAYS_INLINE inline
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
