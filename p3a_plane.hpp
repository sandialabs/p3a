#pragma once

#include "p3a_quantity.hpp"

namespace p3a {

template <class T>
class plane {
  vector3<adimensional_quantity<T>> m_normal;
  length_quantity<T> m_offset;
 public:
  P3A_ALWAYS_INLINE inline
  plane() = default;
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
  length_quantity<T> distance(
      position_quantity<T> const& point) const
  {
    return m_offset + dot_product(m_normal, point);
  }
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
  plane(
      vector3<adimensional_quantity<T>> const& normal_arg,
      length_quantity<T> const& offset_arg)
    :m_normal(normal_arg)
    ,m_offset(offset_arg)
  {
  }
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
  plane(
      position_quantity<T> const& origin_arg,
      vector3<adimensional_quantity<T>> const& normal_arg)
    :plane(normal_arg, -dot_product(normal_arg, origin_arg))
  {
  }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
  vector3<adimensional_quantity<T>> const& normal() const
  {
    return m_normal;
  }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
  length_quantity<T> const& offset() const
  {
    return m_offset;
  }
};

}
