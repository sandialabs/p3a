#pragma once

// functions related to Lie algebras of tensors,
// in particular the tensor logarithm and exponential

#include "p3a_eigen.hpp"
#include "p3a_svd.hpp"
#include "p3a_skew3x3.hpp"
#include "p3a_scaled_identity3x3.hpp"

namespace p3a {

template <class T>
[[nodiscard]] P3A_HOST_DEVICE inline
diagonal3x3<T> logarithm(diagonal3x3<T> const& m)
{
  return diagonal3x3<T>(
      p3a::log(m.xx()),
      p3a::log(m.yy()),
      p3a::log(m.zz()));
}

/* Polar Decomposition:
 * a = u * p
 * (u is unitary, p is symmetric positive definite)
 * Singular Value Decomposition:
 * a = w * s * vt
 * (w and vt are unitary matrices, s is diagonal)
 * Relation between the two:
 * a = w * vt * v * s * vt
 * a = (w * vt) * (v * s * vt)
 * u = w * vt
 * p = v * s * vt
 * Logarithm of (u) is converting it to axis-angle
 * log(p) = v * log(s) * vt
 * since v and thus vt are unitary
 */

template <class T>
[[nodiscard]] P3A_HOST_DEVICE inline
symmetric3x3<T> spd_exponential(symmetric3x3<T> const& log_m)
{
  diagonal3x3<T> l;
  matrix3x3<T> q;
  eigendecompose(log_m, l, q);
  diagonal3x3<T> const exp_l(p3a::exp(l.xx()), p3a::exp(l.yy()), p3a::exp(l.zz()));
  return multiply_a_b_at(q, exp_l);
}

template <class T>
[[nodiscard]] P3A_HOST_DEVICE inline
symmetric3x3<T> spd_logarithm(symmetric3x3<T> const& exp_m)
{
  diagonal3x3<T> l;
  matrix3x3<T> q;
  eigendecompose(exp_m, l, q);
  diagonal3x3<T> const log_l(
      p3a::log(l.xx()),
      p3a::log(l.yy()),
      p3a::log(l.zz()));
  return multiply_a_b_at(q, log_l);
}

enum class polar_errc {
  success,
  singular,
  no_converge
};

template <class T>
[[nodiscard]] P3A_HOST_DEVICE inline
polar_errc polar_rotation(
    matrix3x3<T> const& F,
    matrix3x3<T>& R,
    const int maxit=200)
{
  // Computes the tensor [R] from the polar decompositon (a special case of
  // singular value decomposition),
  //
  //            F = RU = VR
  //
  // for a 3x3 invertible matrix F. Here, R is an orthogonal matrix and U and V
  // are symmetric positive definite matrices.
  //
  // This routine determines only [R].
  //
  // After calling this routine, you can obtain [U] or [V] by
  //       [U] = [R]^T [F]          and        [V] = [F] [R]^T
  //
  // Returns a proper rotation if det[F]>0, or an improper orthogonal tensor if
  // det[F]<0.
  //
  // This routine uses an iterative algorithm, but the iterations are continued
  // until the error is minimized relative to machine precision.  Therefore,
  // this algorithm should be as accurate as any purely analytical method. In
  // fact, this algorithm has been demonstrated to be MORE accurate than
  // analytical solutions because it is less vulnerable to round-off errors.
  //
  // Reference for scaling method:
  // Brannon, R M (2018) "Rotation, Reflection, and Frame Changes"
  // http://dx.doi.org/10.1088/978-0-7503-1454-1
  //
  // Reference for fixed point iterator:
  // Bjorck, A. and Bowie, C. (1971) "An iterative algorithm for computing the
  // best estimate of an orthogonal matrix." SIAM J.  Numer. Anal., vol 8, pp.
  // 358-364.
  //
  // Implementation inspired by the routine polarDecompositionRMB in the Uintah
  // MPM framework.  There, it was found this that algorithm was faster and more
  // robust than other analytic or iterative methods.
  matrix3x3<T> const identity{
    T(1.0), T(0.0), T(0.0),
    T(0.0), T(1.0), T(0.0),
    T(0.0), T(0.0), T(1.0)
  };
  // @daibane: interestingly, tests fail if I use the following instead of the
  // identity matrix defined above.
  //auto const identity = scaled_identity3x3(T(1.0));
  auto const det = determinant(F);
  if (det <= T(0.0)) {
    return polar_errc::singular;
  }
  auto E = transpose(F) * F;
  // To guarantee convergence, scale [F] by multiplying it by
  // Sqrt[3]/magnitude[F]. The rotation for any positive multiple of [F] is the
  // same as the rotation for [F]. Scaling [F] by a factor sqrt(3)/mag[F]
  // requires replacing the previously computed [C] matrix by a factor
  // 3/squareMag[F], where squareMag[F] is most efficiently computed by
  // trace[C].  Complete computation of [E]=(1/2)([C]-[I]) with [C] now being
  // scaled.
  T scale = T(3.0) / trace(E);
  E = (E * scale - identity) * T(0.5);
  // First guess for [R] equal to the scaled [F] matrix,
  // [A]=Sqrt[3]F/magnitude[F]
  scale = p3a::sqrt(scale);
  auto A = scale * F;
  // The matrix [A] equals the rotation if and only if [E] equals [0]
  T err1 = E.xx() * E.xx() + E.yy() * E.yy() + E.zz() * E.zz()
         + T(2.0) * (E.xy() * E.xy() + E.yz() * E.yz() + E.zx() * E.zx());
  // Whenever the stretch tensor is isotropic the scaling alone is sufficient to
  // get rotation.
  if (err1 + T(1.0) == T(1.0)) {
    R = A;
    return polar_errc::success;
  }
  matrix3x3<T> X;
  for (int it=0; it<maxit; it++)
  {
    X = A * (identity - E);
    A = X;
    E = (transpose(A) * A - identity) * T(0.5);
    T err2 = E.xx() * E.xx() + E.yy() * E.yy() + E.zz() * E.zz()
           + T(2.0) * (E.xy() * E.xy() + E.yz() * E.yz() + E.zx() * E.zx());
    // If new error is smaller than old error, then keep on iterating.  If new
    // error equals or exceeds old error, we have reached machine precision
    // accuracy.
    if (err2 >= err1 || err2 + T(1.0) == T(1.0))
    {
      R = A;
      return polar_errc::success;
    }
    err1 = err2;
  }
  return polar_errc::no_converge;
}

template <class T>
[[nodiscard]] P3A_HOST_DEVICE inline
polar_errc decompose_polar_right(
    matrix3x3<T> const& input,
    matrix3x3<T>& rotation,
    symmetric3x3<T>& right_stretch,
    const int maxit=200)
{
  polar_errc const e = polar_rotation(input, rotation, maxit);
  if (e != polar_errc::success) return e;
  right_stretch = symmetric(transpose(rotation) * input);
  return polar_errc::success;
}

template <class T>
[[nodiscard]] P3A_HOST_DEVICE inline
polar_errc decompose_polar_left(
    matrix3x3<T> const& input,
    symmetric3x3<T>& left_stretch,
    matrix3x3<T>& rotation,
    const int maxit=200)
{
  polar_errc const e = polar_rotation(input, rotation, maxit);
  if (e != polar_errc::success) return e;
  left_stretch = symmetric(input * transpose(rotation));
  return polar_errc::success;
}

// Project to O(N) (Orthogonal Group) using a Newton-type algorithm.
// See Higham's Functions of Matrices p210 [2008]
// \param A tensor (often a deformation-gradient-like tensor)
// \return \f$ R = \argmin_Q \|A - Q\|\f$
// This algorithm projects a given tensor in GL(N) to O(N).
// The rotation/reflection obtained through this projection is
// the orthogonal component of the real polar decomposition
template <typename T>
[[nodiscard]] P3A_HOST_DEVICE inline auto
polar_rotation(matrix3x3<T> const& A)
{
  auto const dim       = 3.0;
  auto       scale     = true;
  auto const tol_scale = 0.01;
  auto const tol_conv  = p3a::sqrt(dim) * epsilon_value<T>();
  auto       X         = A;
  auto       gamma     = 2.0;
  auto const max_iter  = 128;
  auto       num_iter  = 0;
  while (num_iter < max_iter) {
    auto const Y  = inverse_full_pivot(X);
    auto       mu = 1.0;
    if (scale == true) {
      mu = (norm_1(Y) * norm_infinity(Y)) / (norm_1(X) * norm_infinity(X));
      mu = p3a::sqrt(p3a::sqrt(mu));
    }
    auto const Z     = 0.5 * (mu * X + transpose(Y) / mu);
    auto const D     = Z - X;
    auto const delta = norm(D) / norm(Z);
    if (scale == true && delta < tol_scale) {
      scale = false;
    }
    auto const end_iter = norm(D) <= p3a::sqrt(tol_conv) || (delta > 0.5 * gamma && scale == false);
    X                   = Z;
    gamma               = delta;
    if (end_iter == true) {
      break;
    }
    num_iter++;
  }
  return X;
}

}
