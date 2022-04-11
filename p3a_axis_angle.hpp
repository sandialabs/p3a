#pragma once

#include "p3a_vector3.hpp"

namespace p3a {

/* this class represents a 3D rotation as the
 * unit vector representing the axis of rotation
 * multiplied by the angle of rotation in radians
 */

template <class T>
class axis_angle {
  vector3<T> m_vector;
 public:
  P3A_ALWAYS_INLINE axis_angle() = default;

  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  axis_angle(vector3<T> const& vector_arg)
    :m_vector(vector_arg)
  {}

/* Markley, F. Landis.
   "Unit quaternion from rotation matrix."
   Journal of guidance, control, and dynamics 31.2 (2008): 440-442.
   Modified Shepperd's algorithm to handle input
   tensors that may not be exactly orthogonal */
// logarithm of a rotation tensor in Special Orthogonal Group(3), as the
// the axis of rotation times the angle of rotation.

  P3A_HOST P3A_DEVICE inline
  axis_angle(matrix3x3<T> const& R)
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
    if (divisor <= epsilon) {
      m_vector = vector3<T>::zero();
    } else {
      auto const factor = T(2.0) * arccos(q0) / divisor;
      m_vector = vector3<T>(q1, q2, q3) * factor;
    }
  }

  [[nodiscard]] P3A_HOST P3A_DEVICE inline
  matrix3x3<T> tensor() const
  {
    auto const halfnorm = T(0.5) * magnitude(m_vector);
    auto const temp = T(0.5) * sin_x_over_x(halfnorm);
    auto const qv = temp * m_vector;
    auto const qs = cosine(halfnorm);
    return T(2.0) * outer_product(qv) +
           T(2.0) * qs * cross_product_matrix(qv) +
           (T(2.0) * square(qs) - T(1.0)) * identity3x3;
  }

  [[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline constexpr
  vector3<T> const& vector() const { return m_vector; }

  [[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE static constexpr
  axis_angle zero()
  {
    return axis_angle(vector3<T>::zero());
  }
};

template <class T>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
axis_angle<T> load_axis_angle(
    T const* ptr,
    int stride,
    int offset)
{
  return axis_angle<T>(load_vector3(ptr, stride, offset));
}

template <class T>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
void store(
    axis_angle<T> const& aa,
    T* ptr,
    int stride,
    int offset)
{
  store(aa.vector(), ptr, stride, offset);
}

template <class T, class Mask>
[[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
std::enable_if_t<!std::is_same_v<Mask, bool>, axis_angle<T>>
condition(
    Mask const& a,
    axis_angle<T> const& b,
    axis_angle<T> const& c)
{
  return axis_angle<T>(condition(a, b.vector(), c.vector()));
}

}
