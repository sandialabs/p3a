#pragma once

#include "p3a_vector3.hpp"
#include "p3a_functions.hpp"

namespace p3a {

template <class T>
class box3 {
  vector3<T> m_lower;
  vector3<T> m_upper;
 public:
  CPL_ALWAYS_INLINE box3() = default;
  CPL_ALWAYS_INLINE constexpr explicit
  box3(vector3<T> const& upper_in)
    :m_lower(vector3<T>::zero())
    ,m_upper(upper_in)
  {}
  CPL_ALWAYS_INLINE constexpr explicit
  box3(vector3<T> const& lower_in, vector3<T> const& upper_in)
    :m_lower(lower_in)
    ,m_upper(upper_in)
  {}
  CPL_ALWAYS_INLINE constexpr
  box3(T const& a, T const& b, T const& c)
    :box3(vector3<T>(a, b, c))
  {}
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  vector3<T> extents() const
  {
    return m_upper - m_lower;
  }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  auto volume() const
  {
    return extents().volume();
  }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  vector3<T> const& lower() const
  {
    return m_lower;
  }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  vector3<T> const& upper() const
  {
    return m_upper;
  }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  vector3<T>& lower()
  {
    return m_lower;
  }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  vector3<T>& upper()
  {
    return m_upper;
  }
  [[nodiscard]] CPL_ALWAYS_INLINE constexpr
  bool operator==(box3 const& other) const
  {
    return m_lower == other.m_lower &&
           m_upper == other.m_upper;
  }
};

template <class T>
[[nodiscard]] CPL_ALWAYS_INLINE constexpr
box3<T> intersect(box3<T> const& a, box3<T> const& b)
{
  auto const lower = vector3<T>(
      maximum(a.lower().x(), b.lower().x()),
      maximum(a.lower().y(), b.lower().y()),
      maximum(a.lower().z(), b.lower().z()));
  auto const upper = vector3<T>(
      maximum(lower.x(), minimum(a.upper().x(), b.upper().x())),
      maximum(lower.y(), minimum(a.upper().y(), b.upper().y())),
      maximum(lower.z(), minimum(a.upper().z(), b.upper().z())));
  return box3<T>(lower, upper);
}

}
