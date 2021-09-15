#pragma once

// functions related to Lie algebras of tensors,
// in particular the tensor logarithm and exponential

#include "p3a_eigen.hpp"
#include "p3a_svd.hpp"
#include "p3a_skew3x3.hpp"
#include "p3a_scaled_identity3x3.hpp"

namespace p3a {

/* Markley, F. Landis.
   "Unit quaternion from rotation matrix."
   Journal of guidance, control, and dynamics 31.2 (2008): 440-442.

   Modified Shepperd's algorithm to handle input
   tensors that may not be exactly orthogonal */

// logarithm of a rotation tensor in Special Orthogonal Group(3), as the
// the axis of rotation times the angle of rotation.

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE inline
vector3<T> axis_angle_from_tensor(matrix3x3<T> const& R)
{
  T const trR = trace(R);
  T maxm = trR;
  int maxi = 3;
  if (R.xx() > maxm) {
    maxm = R.xx();
    maxi = 0;
  }
  if (R.yy() > maxm) {
    maxm = R.yy();
    maxi = 1;
  }
  if (R.zz() > maxm) {
    maxm = R.zz();
    maxi = 2;
  }
  T q0, q1, q2, q3; // quaternion components
  if (maxi == 0) {
    q1 = T(1.0) + R.xx() - R.yy() - R.zz();
    q2 = R.xy() + R.yx();
    q3 = R.xz() + R.zx();
    q0 = R.zy() - R.yz();
  } else if (maxi == 1) {
    q1 = R.yx() + R.xy();
    q2 = T(1.0) + R.yy() - R.zz() - R.xx();
    q3 = R.yz() + R.zy();
    q0 = R.xz() - R.zx();
  } else if (maxi == 2) {
    q1 = R.zx() + R.xz();
    q2 = R.zy() + R.yz();
    q3 = T(1.0) + R.zz() - R.xx() - R.yy();
    q0 = R.yx() - R.xy();
  } else if (maxi == 3) {
    q1 = R.zy() - R.yz();
    q2 = R.xz() - R.zx();
    q3 = R.yx() - R.xy();
    q0 = T(1.0) + trR;
  }
  auto const qnorm =
    square_root(
        square(q0) +
        square(q1) +
        square(q2) +
        square(q3));
  q0 /= qnorm;
  q1 /= qnorm;
  q2 /= qnorm;
  q3 /= qnorm;
  // convert quaternion to axis-angle
  auto const divisor = square_root(T(1.0) - square(q0));
  auto constexpr epsilon = epsilon_value<T>();
  if (divisor < epsilon) {
    return vector3<T>::zero();
  } else {
    auto const factor = T(2.0) * arccos(q0) / divisor;
    return vector3<T>(
        q1 * factor,
        q2 * factor,
        q3 * factor);
  }
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE inline
matrix3x3<T> tensor_from_axis_angle(vector3<T> const& aa)
{
  auto const halfnorm = T(0.5) * length(aa);
  auto const temp = T(0.5) * sin_x_over_x(halfnorm);
  auto const qv = temp * aa;
  auto const qs = std::cos(halfnorm);
  return T(2.0) * outer_product(qv) +
         T(2.0) * qs * cross_product_matrix(qv) +
         (T(2.0) * square(qs) - T(1.0)) * identity3x3;
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
matrix3x3<T> pack_polar(
    symmetric3x3<T> const& spd,
    vector3<T> const& aa)
{
  return matrix3x3<T>(
    spd.xx(),
    spd.xy(),
    spd.xz(),
    aa.x(),
    spd.yy(),
    spd.yz(),
    aa.y(),
    aa.z(),
    spd.zz());
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
symmetric3x3<T> unpack_polar_spd(
    matrix3x3<T> const& packed)
{
  return symmetric3x3<T>(
      packed.xx(),
      packed.xy(),
      packed.xz(),
      packed.yy(),
      packed.yz(),
      packed.zz());
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
vector3<T> unpack_polar_axis_angle(
    matrix3x3<T> const& packed)
{
  return vector3<T>(
      packed.yx(),
      packed.zx(),
      packed.zy());
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE inline
diagonal3x3<T> logarithm(diagonal3x3<T> const& m)
{
  return diagonal3x3<T>(
      std::log(m.xx()),
      std::log(m.yy()),
      std::log(m.zz()));
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
[[nodiscard]] P3A_HOST P3A_DEVICE inline
matrix3x3<T> polar_logarithm(matrix3x3<T> const& a)
{
  matrix3x3<T> w, vt;
  diagonal3x3<T> s;
  decompose_singular_values(a, w, s, vt);
  auto const u = w * vt;
  auto const log_u = axis_angle_from_tensor(u);
  auto const log_s = logarithm(s);
  auto const log_p = multiply_at_b_a(vt, log_s);
  return pack_polar(log_p, log_u);
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE inline
symmetric3x3<T> spd_exponential(symmetric3x3<T> const& log_m)
{
  diagonal3x3<T> l;
  matrix3x3<T> q;
  eigendecompose(log_m, l, q);
  diagonal3x3<T> const exp_l(
      std::exp(l.xx()),
      std::exp(l.yy()),
      std::exp(l.zz()));
  return multiply_a_b_at(q, exp_l);
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE inline
symmetric3x3<T> spd_logarithm(symmetric3x3<T> const& exp_m)
{
  diagonal3x3<T> l;
  matrix3x3<T> q;
  eigendecompose(exp_m, l, q);
  diagonal3x3<T> const log_l(
      std::log(l.xx()),
      std::log(l.yy()),
      std::log(l.zz()));
  return multiply_a_b_at(q, log_l);
}

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE inline
matrix3x3<T> polar_exponential(matrix3x3<T> const& packed)
{
  auto const log_u = unpack_polar_axis_angle(packed);
  auto const u = tensor_from_axis_angle(log_u);
  auto const log_p = unpack_polar_spd(packed);
  auto const p = spd_exponential(log_p);
  auto const a = u * p;
  return a;
}

enum class polar_errc {
  success,
  singular,
  no_converge
};

template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE inline
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
  scale = std::sqrt(scale);
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
[[nodiscard]] P3A_HOST P3A_DEVICE inline
polar_errc decompose_polar(
    matrix3x3<T> const& F,
    matrix3x3<T>& R,
    symmetric3x3<T>& U,
    const int maxit=200)
{
  polar_errc const e = polar_rotation(F, R, maxit);
  if (e != polar_errc::success) return e;
  U = symmetric(transpose(R) * F);
  return polar_errc::success;
}


}
