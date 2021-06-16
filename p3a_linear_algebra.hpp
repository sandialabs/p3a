#pragma once

#include <functional>
#include "p3a_reduce.hpp"

namespace p3a {

class dense_matrix {
  int m_row_count;
  int m_column_count;
  dynamic_array<double> m_storage;
 public:
  dense_matrix()
    :m_row_count(0)
    ,m_column_count(0)
  {}
  dense_matrix(dense_matrix const&) = default;
  dense_matrix& operator=(dense_matrix const&) = default;
  dense_matrix(dense_matrix&&) = default;
  dense_matrix& operator=(dense_matrix&&) = default;
  dense_matrix(int row_count_arg, int column_count_arg)
    :m_row_count(row_count_arg)
    ,m_column_count(column_count_arg)
    ,m_storage(row_count_arg * column_count_arg)
  {
  }
  void zero();
  void resize(int new_row_count, int new_column_count);
  [[nodiscard]] P3A_ALWAYS_INLINE inline constexpr
  int row_count() const { return m_row_count; }
  [[nodiscard]] P3A_ALWAYS_INLINE inline constexpr
  int column_count() const { return m_column_count; }
  [[nodiscard]] P3A_ALWAYS_INLINE inline constexpr
  double& operator()(int i, int j) { return m_storage[i * m_column_count + j]; }
  [[nodiscard]] P3A_ALWAYS_INLINE inline constexpr
  double const& operator()(int i, int j) const { return m_storage[i * m_column_count + j]; }
};

void axpy(double a, dense_matrix const& x, dense_matrix const& y, dense_matrix& result);
void multiply(dense_matrix const& a, dense_matrix const& b, dense_matrix& result);
void solve(dense_matrix& a, dense_matrix& b, dense_matrix& x);

}
