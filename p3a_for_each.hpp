#pragma once

namespace p3a {

template <class Functor>
CPL_NEVER_INLINE void for_each(
    execution::serial_policy,
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
    execution::local_policy,
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
    execution::serial_policy,
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
    execution::local_policy,
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
