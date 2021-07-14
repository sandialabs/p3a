#pragma once

// functions related to Lie algebras of tensors,
// in particular the tensor logarithm and exponential

#include "p3a_eigen.hpp"
#include "p3a_svd.hpp"

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
  return T(2.0) * outer_product(qv, qv) +
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
  auto const U = w * vt;
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
      std::exp(l.x()),
      std::exp(l.y()),
      std::exp(l.z()));
  return multiply_a_b_at(q, exp_l);
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

}
