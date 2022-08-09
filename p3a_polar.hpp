#pragma once

// functions for polar decomposition of 3x3 tensors

#include "p3a_matrix3x3.hpp"

namespace p3a {

enum class polar_errc {
  success,
  singular,
  no_converge
};

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

template <class T>
[[nodiscard]] P3A_HOST_DEVICE inline
polar_errc polar_rotation_fast(
    matrix3x3<T> const& F,
    matrix3x3<T>& R,
    const int maxit=200)
{
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
  T          gamma     = 2.0;
  auto const max_iter  = 128;
  auto       num_iter  = 0;
  while (num_iter < max_iter) {
    auto const Y  = inverse_full_pivot(X);
    T          mu = 1.0;
    if (scale == true) {
      mu = (norm_1(Y) * norm_infinity(Y)) / (norm_1(X) * norm_infinity(X));
      mu = sqrt(sqrt(mu));
    }
    auto const Z     = 0.5 * (mu * X + transpose(Y) / mu);
    auto const D     = Z - X;
    auto const delta = norm(D) / norm(Z);
    if (scale == true && delta < tol_scale) {
      scale = false;
    }
    auto const end_iter = norm(D) <= sqrt(tol_conv) || (delta > 0.5 * gamma && scale == false);
    X                   = Z;
    gamma               = delta;
    if (end_iter == true) {
      break;
    }
    num_iter++;
  }
  return X;
}

template <class T>
P3A_HOST_DEVICE inline
void decompose_polar_right(
    matrix3x3<T> const& input,
    matrix3x3<T>& rotation,
    symmetric3x3<T>& right_stretch)
{
  rotation = polar_rotation(input);
  right_stretch = symmetric_part(transpose(rotation) * input);
}

template <class T>
P3A_HOST_DEVICE inline
void decompose_polar_left(
    matrix3x3<T> const& input,
    symmetric3x3<T>& left_stretch,
    matrix3x3<T>& rotation)
{
  rotation = polar_rotation(input);
  left_stretch = symmetric_part(input * transpose(rotation));
}

}
