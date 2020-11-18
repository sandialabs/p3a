#pragma once

namespace p3a {

template <class Functor>
CPL_NEVER_INLINE void for_each(
    serial_execution,
    grid3 grid,
    Functor functor)
{
  for (int k = 0; k < grid.extents().z(); ++k) {
    for (int j = 0; j < grid.extents().y(); ++j) {
      for (int i = 0; i < grid.extents().x(); ++i) {
        functor(vector3<int>(i, j, k));
      }
    }
  }
}

template <class Functor>
CPL_ALWAYS_INLINE constexpr void for_each(
    local_execution,
    grid3 const& grid,
    Functor const& functor)
{
  for (int k = 0; k < grid.extents().z(); ++k) {
    for (int j = 0; j < grid.extents().y(); ++j) {
      for (int i = 0; i < grid.extents().x(); ++i) {
        functor(vector3<int>(i, j, k));
      }
    }
  }
}

template <class Functor>
CPL_NEVER_INLINE void for_each(
    serial_execution,
    subgrid3 subgrid,
    Functor functor)
{
  for (int k = subgrid.lower().z(); k < subgrid.upper().z(); ++k) {
    for (int j = subgrid.lower().y(); j < subgrid.upper().y(); ++j) {
      for (int i = subgrid.lower().x(); i < subgrid.upper().x(); ++i) {
        functor(vector3<int>(i, j, k));
      }
    }
  }
}

template <class Functor>
CPL_ALWAYS_INLINE constexpr void for_each(
    local_execution,
    subgrid3 const& subgrid,
    Functor const& functor)
{
  for (int k = subgrid.lower().z(); k < subgrid.upper().z(); ++k) {
    for (int j = subgrid.lower().y(); j < subgrid.upper().y(); ++j) {
      for (int i = subgrid.lower().x(); i < subgrid.upper().x(); ++i) {
        functor(vector3<int>(i, j, k));
      }
    }
  }
}

}
