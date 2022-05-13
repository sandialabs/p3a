#pragma once

#include "p3a_static_matrix.hpp"
#include "p3a_constants.hpp"
#include "p3a_symmetric3x3.hpp"
#include "p3a_matrix3x3.hpp"

namespace p3a {

template <class T, int N>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
T norm(static_matrix<T, N, N> const& a)
{
  T result(0);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      result += square(a(i, j));
    }
  }
  return square_root(result);
}

template <class T, int N>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
T off_diagonal_norm(static_matrix<T, N, N> const& a)
{
  T result(0);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (i != j) {
        result += square(a(i, j));
      }
    }
  }
  return square_root(result);
}

template <class T, int N>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
void maximum_off_diagonal_indices(
    static_matrix<T, N, N> const& a,
    int& p,
    int& q)
{
  using std::abs;
  p = 0;
  q = 0;
  T s = -1.0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (i != j) {
        T const s2 = abs(a(i, j));
        if (s2 > s) {
          p = i;
          q = j;
          s = s2;
        }
      }
    }
  }
  if (q < p) p3a::swap(p, q);
}

// Symmetric Schur algorithm for R^2.
// \param \f$ A = [f, g; g, h] \in S(2) \f$
// \return \f$ c, s \rightarrow [c, -s; s, c]\f diagonalizes A$
template <class T>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
void symmetric_schur(
    T const& f,
    T const& g,
    T const& h,
    T& c,
    T& s)
{
  c = 1.0;
  s = 0.0;
  if (g != 0.0) {
    T t = (h - f) / (2.0 * g);
    if (t >= 0.0) {
      t = 1.0 / (square_root(1.0 + square(t)) + t);
    } else {
      t = -1.0 / (square_root(1.0 + square(t)) - t);
    }
    c = 1.0 / square_root(1.0 + square(t));
    s = t * c;
  }
}

/* Apply Givens-Jacobi rotation on the left */
template <class T, int N>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
void rotate_givens_left(
    T const& c,
    T const& s,
    int const i,
    int const k,
    static_matrix<T, N, N>& a)
{
  for (int j = 0; j < N; ++j) {
    T const t1 = a(i, j);
    T const t2 = a(k, j);
    a(i, j) = c * t1 - s * t2;
    a(k, j) = s * t1 + c * t2;
  }
}

/* Apply Givens-Jacobi rotation on the right */
template <class T, int N>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
void rotate_givens_right(
    T const& c,
    T const& s,
    int const i,
    int const k,
    static_matrix<T, N, N>& a)
{
  for (int j = 0; j < N; ++j) {
    T const t1 = a(j, i);
    T const t2 = a(j, k);
    a(j, i) = c * t1 - s * t2;
    a(j, k) = s * t1 + c * t2;
  }
}

template <class T, int N>
P3A_HOST_DEVICE inline
void eigendecompose(
    static_matrix<T, N, N>& a,
    static_matrix<T, N, N>& q,
    T const& tolerance)
{
  q.assign_identity();
  int constexpr maximum_iteration_count = (5 * N * N) / 2;
  for (int iteration = 0; iteration < maximum_iteration_count; ++iteration) {
    T const odn = off_diagonal_norm(a);
    if (odn <= tolerance) break;
    int i, j;
    maximum_off_diagonal_indices(a, i, j);
    T const f = a(i, i);
    T const g = a(i, j);
    T const h = a(j, j);
    T c, s;
    symmetric_schur(f, g, h, c, s);
    rotate_givens_left(c, s, i, j, a);
    rotate_givens_right(c, s, i, j, a);
    rotate_givens_right(c, s, i, j, q);
  }
}

template <class T, int N>
P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
T eigen_tolerance(
    static_matrix<T, N, N> const& a)
{
  T constexpr epsilon = epsilon_value<T>();
  T const tolerance = epsilon * norm(a);
  return tolerance;
}

template <class T, int N>
P3A_HOST_DEVICE inline
void eigendecompose(
    static_matrix<T, N, N>& a,
    static_matrix<T, N, N>& q)
{
  eigendecompose(a, q, eigen_tolerance(a));
}

template <class T>
P3A_HOST_DEVICE inline
void eigendecompose(
    symmetric3x3<T> const& a,
    diagonal3x3<T>& l,
    matrix3x3<T>& q)
{
  static_matrix<T, 3, 3> a2;
  static_matrix<T, 3, 3> q2;
  a2(0, 0) = a.xx();
  a2(0, 1) = a.xy();
  a2(0, 2) = a.xz();
  a2(1, 0) = a.yx();
  a2(1, 1) = a.yy();
  a2(1, 2) = a.yz();
  a2(2, 0) = a.zx();
  a2(2, 1) = a.zy();
  a2(2, 2) = a.zz();
  eigendecompose(a2, q2);
  l.xx() = a2(0, 0);
  l.yy() = a2(1, 1);
  l.zz() = a2(2, 2);
  q.xx() = q2(0, 0);
  q.xy() = q2(0, 1);
  q.xz() = q2(0, 2);
  q.yx() = q2(1, 0);
  q.yy() = q2(1, 1);
  q.yz() = q2(1, 2);
  q.zx() = q2(2, 0);
  q.zy() = q2(2, 1);
  q.zz() = q2(2, 2);
}

}
