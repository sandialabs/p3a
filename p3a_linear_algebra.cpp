#include "p3a_linear_algebra.hpp"

#include "p3a_counting_iterator.hpp"

namespace p3a {

void dense_matrix::zero()
{
  for (int i = 0; i < m_row_count; ++i) {
    for (int j = 0; j < m_column_count; ++j) {
      operator()(i, j) = 0.0;
    }
  }
}

void dense_matrix::resize(int new_row_count, int new_column_count)
{
  if (new_row_count != m_row_count || new_column_count != m_column_count) {
    m_storage.resize(0);
    m_storage.resize(new_row_count * new_column_count);
  }
}

void axpy(double a, dense_matrix const& x, dense_matrix const& y, dense_matrix& result)
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

void multiply(dense_matrix const& a, dense_matrix const& b, dense_matrix& result)
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
      double value = 0.0;
      for (int k = 0; k < o; ++k) {
        value += a(i, k) * b(k, j);
      }
      result(i, j) = value;
    }
  }
}

P3A_ALWAYS_INLINE inline
void swap_rows(dense_matrix& A, int a, int b) {
  for (int j = 0; j < A.column_count(); ++j) {
    std::swap(A(a, j), A(b, j));
  }
}

P3A_NEVER_INLINE
void gaussian_elimination(
    dense_matrix& a,
    dense_matrix& b)
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
    double max_magnitude = std::abs(a(i_max, k));
    for (int i = h + 1; i < m; ++i) {
      double const magnitude = std::abs(a(i, k));
      if (magnitude > max_magnitude) {
        i_max = i;
        max_magnitude = magnitude;
      }
    }
    if (a(i_max, k) == double(0.0)) {
      // no pivot in this column, pass to next column
      k = k + 1;
    } else {
      swap_rows(a, h, i_max);
      std::swap(b(h, 0), b(i_max, 0));
      // for all rows below the pivot
      for (int i = h + 1; i < m; ++i) {
        double const f = a(i, k) / a(h, k);
        // fill with zeros the lower part of the pivot column
        a(i, k) = double(0.0);
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

void back_substitution(
    dense_matrix const& U,
    dense_matrix const& y,
    dense_matrix& x) {
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
    double sum = y(i, 0);
    for (int j = n - 1; j > i; --j) {
      sum = sum - U(i, j) * x(j, 0);
    }
    x(i, 0) = sum / U(i, i);
  }
}

void solve(dense_matrix& a, dense_matrix& b, dense_matrix& x)
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

P3A_NEVER_INLINE double dot_product(
    reproducible_floating_point_adder& adder,
    device_array<double> const& a,
    device_array<double> const& b)
{
  using size_type = device_array<double>::size_type;
  auto const a_ptr = a.cbegin();
  auto const b_ptr = b.cbegin();
  return adder.transform_reduce(
      counting_iterator<size_type>(0),
      counting_iterator<size_type>(a.size()),
  [=] P3A_DEVICE (size_type i) P3A_ALWAYS_INLINE {
    return a_ptr[i] * b_ptr[i];
  });
}

P3A_NEVER_INLINE void axpy(
    double a,
    device_array<double> const& x,
    device_array<double> const& y,
    device_array<double>& result)
{
  using size_type = device_array<double>::size_type;
  auto const x_ptr = x.cbegin();
  auto const y_ptr = y.cbegin();
  auto const result_ptr = result.begin();
  for_each(device,
      counting_iterator<size_type>(0),
      counting_iterator<size_type>(x.size()),
  [=] P3A_DEVICE (size_type i) P3A_ALWAYS_INLINE {
    result_ptr[i] = a * x_ptr[i] + y_ptr[i];
  });
}

int conjugate_gradient_solver::solve(
      Ax_functor_type const& A_action,
      b_functor_type const& b_functor,
      device_array<double>& x,
      double relative_tolerance)
{
  device_array<double>& r = this->m_r;
  device_array<double>& p = this->m_p;
  device_array<double>& Ap = this->m_Ap;
  device_array<double>& Ax = this->m_r;
  device_array<double>& b = this->m_Ap;
  b_functor(b);
  double const b_dot_b = dot_product(m_adder, b, b);
  double const b_magnitude = square_root(b_dot_b);
  double const absolute_tolerance = b_magnitude * relative_tolerance;
  A_action(x, Ax);
  axpy(-1.0, Ax, b, r); // r = A * x - b
  copy(device, r.cbegin(), r.cend(), p.begin()); // p = r
  double r_dot_r_old = dot_product(m_adder, r, r);
  double residual_magnitude = square_root(r_dot_r_old);
  if (residual_magnitude <= absolute_tolerance) return 0;
  for (int k = 1; true; ++k) {
    A_action(p, Ap);
    double const pAp = dot_product(m_adder, p, Ap);
    double const alpha = r_dot_r_old / pAp; // alpha = (r^T * r) / (p^T * A * p)
    axpy(alpha, p, x, x); // x = x + alpha * p
    axpy(-alpha, Ap, r, r); // r = r - alpha * (A * p)
    double const r_dot_r_new = dot_product(m_adder, r, r);
    residual_magnitude = square_root(r_dot_r_old);
    if (residual_magnitude <= absolute_tolerance) {
      return k;
    }
    double const beta = r_dot_r_new / r_dot_r_old;
    axpy(beta, p, r, p); // p = r + beta * p
    r_dot_r_old = r_dot_r_new;
  }
}

}
