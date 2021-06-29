#pragma once

#include <functional>
#include <stdexcept>

#include "p3a_dynamic_array.hpp"

namespace p3a {

template <
  class T,
  class Allocator = allocator<T>,
  class ExecutionPolicy = serial_execution>
class dynamic_matrix {
  int m_row_count;
  int m_column_count;
  dynamic_array<T, Allocator, ExecutionPolicy> m_storage;
 public:
  dynamic_matrix()
    :m_row_count(0)
    ,m_column_count(0)
  {}
  dynamic_matrix(dynamic_matrix const&) = default;
  dynamic_matrix& operator=(dynamic_matrix const&) = default;
  dynamic_matrix(dynamic_matrix&&) = default;
  dynamic_matrix& operator=(dynamic_matrix&&) = default;
  dynamic_matrix(int row_count_arg, int column_count_arg)
    :m_row_count(row_count_arg)
    ,m_column_count(column_count_arg)
    ,m_storage(row_count_arg * column_count_arg)
  {
  }
  void assign_zero() {
    for (int i = 0; i < m_row_count; ++i) {
      for (int j = 0; j < m_column_count; ++j) {
        operator()(i, j) = T(0);
      }
    }
  }
  void resize(int new_row_count, int new_column_count)
  {
    if (new_row_count != m_row_count || new_column_count != m_column_count) {
      m_storage.resize(0);
      m_storage.resize(new_row_count * new_column_count);
      m_row_count = new_row_count;
      m_column_count = new_column_count;
    }
  }
  [[nodiscard]] P3A_ALWAYS_INLINE inline constexpr
  int row_count() const { return m_row_count; }
  [[nodiscard]] P3A_ALWAYS_INLINE inline constexpr
  int column_count() const { return m_column_count; }
  [[nodiscard]] P3A_ALWAYS_INLINE inline constexpr
  T& operator()(int i, int j) { return m_storage[i * m_column_count + j]; }
  [[nodiscard]] P3A_ALWAYS_INLINE inline constexpr
  T const& operator()(int i, int j) const { return m_storage[i * m_column_count + j]; }
};

template <class T, class Allocator, class ExecutionPolicy>
void axpy(
    T a,
    dynamic_matrix<T, Allocator, ExecutionPolicy> const& x,
    dynamic_matrix<T, Allocator, ExecutionPolicy> const& y,
    dynamic_matrix<T, Allocator, ExecutionPolicy>& result)
{
  int n = x.row_count();
  int m = x.column_count();
  if (n != y.row_count() || m != y.column_count()) {
    throw std::invalid_argument("dense axpy: y wrong size");
  }
  result.resize(n, m);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      result(i, j) = a * x(i, j) + y(i, j);
    }
  }
}

template <class T, class Allocator, class ExecutionPolicy>
void multiply(
    dynamic_matrix<T, Allocator, ExecutionPolicy> const& a,
    dynamic_matrix<T, Allocator, ExecutionPolicy> const& b,
    dynamic_matrix<T, Allocator, ExecutionPolicy>& result)
{
  int const n = a.row_count();
  int const m = b.column_count();
  int const o = a.column_count();
  if (o != b.row_count()) {
    throw std::invalid_argument(
        "dense matrix multiply: LHS columns != RHS rows");
  }
  for (int j = 0; j < m; ++j) {
    for (int i = 0; i < n; ++i) {
      T value(0);
      for (int k = 0; k < o; ++k) {
        value += a(i, k) * b(k, j);
      }
      result(i, j) = value;
    }
  }
}

template <class T, class Allocator, class ExecutionPolicy>
P3A_ALWAYS_INLINE inline
void swap_rows(
    dynamic_matrix<T, Allocator, ExecutionPolicy>& A,
    int a, int b)
{
  for (int j = 0; j < A.column_count(); ++j) {
    std::swap(A(a, j), A(b, j));
  }
}

template <class T, class Allocator, class ExecutionPolicy>
P3A_NEVER_INLINE
void gaussian_elimination(
    dynamic_matrix<T, Allocator, ExecutionPolicy>& a,
    dynamic_matrix<T, Allocator, ExecutionPolicy>& b)
{
  if (a.row_count() != b.row_count()) {
    throw std::invalid_argument(
        "Gaussian elimination: row counts are different");
  }
  int const m = a.row_count();
  int const n = a.column_count();
  int h = 0; // pivot row
  int k = 0; // pivot column
  while ((h < m) && (k < n)) {
    // find the k-th pivot
    int i_max = h;
    T max_magnitude = absolute_value(a(i_max, k));
    for (int i = h + 1; i < m; ++i) {
      T const magnitude = absolute_value(a(i, k));
      if (magnitude > max_magnitude) {
        i_max = i;
        max_magnitude = magnitude;
      }
    }
    if (a(i_max, k) == T(0)) {
      // no pivot in this column, pass to next column
      k = k + 1;
    } else {
      swap_rows(a, h, i_max);
      std::swap(b(h, 0), b(i_max, 0));
      // for all rows below the pivot
      for (int i = h + 1; i < m; ++i) {
        T const f = a(i, k) / a(h, k);
        // fill with zeros the lower part of the pivot column
        a(i, k) = T(0);
        // for all remaining elements in the current row
        for (int j = k + 1; j < n; ++j) {
          a(i, j) = a(i, j) - a(h, j) * f;
        }
        b(i, 0) = b(i, 0) - b(h, 0) * f;
      }
      // increase pivot row and column
      h = h + 1;
      k = k + 1;
    }
  }
}

template <class T, class Allocator, class ExecutionPolicy>
void back_substitution(
    dynamic_matrix<T, Allocator, ExecutionPolicy> const& U,
    dynamic_matrix<T, Allocator, ExecutionPolicy> const& y,
    dynamic_matrix<T, Allocator, ExecutionPolicy>& x) {
  if (U.row_count() != U.column_count()) {
    throw std::invalid_argument(
        "back substitution: U not square");
  }
  if (U.row_count() != x.row_count() ||
      x.column_count() != 1) {
    throw std::invalid_argument(
        "back substitution: x wrong size");
  }
  if (U.row_count() != y.row_count() ||
      y.column_count() != 1) {
    throw std::invalid_argument(
        "back substitution: y wrong size");
  }
  int const n = U.row_count();
  x(n - 1, 0) = y(n - 1, 0) / U(n - 1, n - 1);
  for (int i = n - 2; i >= 0; --i) {
    T sum = y(i, 0);
    for (int j = n - 1; j > i; --j) {
      sum = sum - U(i, j) * x(j, 0);
    }
    x(i, 0) = sum / U(i, i);
  }
}

template <class T, class Allocator, class ExecutionPolicy>
void solve(
    dynamic_matrix<T, Allocator, ExecutionPolicy>& a,
    dynamic_matrix<T, Allocator, ExecutionPolicy>& b,
    dynamic_matrix<T, Allocator, ExecutionPolicy>& x)
{
  if (a.row_count() != b.row_count()) {
    throw std::invalid_argument(
        "direct solve: A rows != b rows");
  }
  if (a.column_count() != x.row_count()) {
    throw std::invalid_argument(
        "direct solve: A columns != x rows");
  }
  if (x.column_count() != 1) {
    throw std::invalid_argument(
        "direct solve: x not vector");
  }
  if (b.column_count() != 1) {
    throw std::invalid_argument(
        "direct solve: b not vector");
  }
  gaussian_elimination(a, b);
  back_substitution(a, b, x);
}

}
