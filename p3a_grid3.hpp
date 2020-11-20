#pragma once

#include "p3a_vector3.hpp"
#include "p3a_box3.hpp"

namespace p3a {

class grid3 {
  vector3<int> m_extents;
 public:
  P3A_ALWAYS_INLINE grid3() = default;
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  grid3(int a, int b, int c)
    :m_extents(a, b, c)
  {}
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  grid3(vector3<int> const& extents_in)
    :m_extents(extents_in)
  {}
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  int index(vector3<int> const& point) const
  {
    // "layout left"
    return (point.z() * m_extents.y() + point.y()) * m_extents.x() + point.x();
  }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  int size() const
  {
    return m_extents.volume();
  }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  vector3<int> const& extents() const
  {
    return m_extents;
  }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  vector3<int>& extents()
  {
    return m_extents;
  }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  bool contains(vector3<int> const& p) const
  {
    return p.x() >= 0 &&
           p.y() >= 0 &&
           p.z() >= 0 &&
           p.x() < m_extents.x() &&
           p.y() < m_extents.y() &&
           p.z() < m_extents.z();
  }
};

class subgrid3 {
  box3<int> m_box;
 public:
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr subgrid3(
      vector3<int> const& lower_in,
      vector3<int> const& upper_in)
    :m_box(lower_in, upper_in)
  {}
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr subgrid3(
      grid3 const& grid_in)
    :subgrid3(vector3<int>::zero(), grid_in.extents())
  {}
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  vector3<int> const& lower() const
  {
    return m_box.lower();
  }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  vector3<int> const& upper() const
  {
    return m_box.upper();
  }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  vector3<int>& lower()
  {
    return m_box.lower();
  }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  vector3<int>& upper()
  {
    return m_box.upper();
  }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  bool operator==(subgrid3 const& other) const
  {
    return m_box == other.m_box;
  }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  bool operator!=(subgrid3 const& other) const
  {
    return !operator==(other);
  }
};

}
