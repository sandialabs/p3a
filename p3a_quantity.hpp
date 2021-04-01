#pragma once

#include "p3a_dimension.hpp"
#include "p3a_vector3.hpp"
#include "p3a_symmetric3x3.hpp"
#include "p3a_matrix3x3.hpp"
#include "p3a_simd.hpp"

namespace p3a {

template <
  class T,
  class Dimension>
class quantity {
  T m_value;
 public:
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  quantity(T const& value_in)
    :m_value(value_in)
  {}
  P3A_ALWAYS_INLINE quantity() = default;
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& value() const
  {
    return m_value;
  }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& value()
  {
    return m_value;
  }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE static constexpr
  quantity zero()
  {
    return quantity(zero_value<T>());
  }
};

template <class T, class Dimension>
struct is_scalar_helper<quantity<T, Dimension>> {
  inline static constexpr bool value = true;
};

template <class T, class Dimension>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
quantity<T, Dimension> operator+(
    quantity<T, Dimension> const& a,
    quantity<T, Dimension> const& b)
{
  return quantity<T, Dimension>(a.value() + b.value());
}

template <class T, class Dimension>
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
quantity<T, Dimension>& operator+=(
    quantity<T, Dimension>& a,
    quantity<T, Dimension> const& b)
{
  a = a + b;
  return a;
}

template <class T, class Dimension>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
quantity<T, Dimension> operator-(
    quantity<T, Dimension> const& a)
{
  return quantity<T, Dimension>(-a.value());
}

template <class T, class Dimension>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
quantity<T, Dimension> operator-(
    quantity<T, Dimension> const& a,
    quantity<T, Dimension> const& b)
{
  return quantity<T, Dimension>(a.value() - b.value());
}

template <class T, class Dimension>
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
quantity<T, Dimension>& operator-=(
    quantity<T, Dimension>& a,
    quantity<T, Dimension> const& b)
{
  a = a - b;
  return a;
}

template <class T, class Dimension>
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
quantity<T, Dimension>& operator*=(
    quantity<T, Dimension>& a,
    quantity<T, adimensional> const& b)
{
  a = a * b;
  return a;
}

template <class T, class ADimension, class BDimension>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator/(
    quantity<T, ADimension> const& a,
    quantity<T, BDimension> const& b)
{
  using result_dimension = dimension_quotient<ADimension, BDimension>;
  return quantity<T, result_dimension>(a.value() / b.value());
}

template <class A, class T, class Dimension>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<A>, quantity<T, dimension_quotient<adimensional, Dimension>>>::type
operator/(
    A const& a,
    quantity<T, Dimension> const& b)
{
  using result_dimension = dimension_quotient<adimensional, Dimension>;
  return quantity<T, result_dimension>(a / b.value());
}

template <class T, class Dimension, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<B>, quantity<T, Dimension>>::type
operator/(
    quantity<T, Dimension> const& a,
    B const& b)
{
  return quantity<T, Dimension>(a.value() / b);
}

template <class T, class Dimension, class B>
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<B>, quantity<T, Dimension>&>::type
operator/=(
    quantity<T, Dimension>& a,
    B const& b)
{
  a = a / b;
  return a;
}

template <class T, class ADimension, class BDimension>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(
    quantity<T, ADimension> const& a,
    quantity<T, BDimension> const& b)
{
  return quantity<T, dimension_product<ADimension, BDimension>>(a.value() * b.value());
}

template <class A, class T, class Dimension>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<A>, quantity<T, Dimension>>::type operator*(
    A const& a,
    quantity<T, Dimension> const& b)
{
  return quantity<T, Dimension>(a * b.value());
}

template <class T, class Dimension, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<B>, quantity<T, Dimension>>::type operator*(
    quantity<T, Dimension> const& a,
    B const& b)
{
  return quantity<T, Dimension>(a.value() * b);
}

template <class T, class Dimension>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator==(
    quantity<T, Dimension> const& a,
    quantity<T, Dimension> const& b)
{
  return a.value() == b.value();
}

template <class T, class Dimension>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator!=(
    quantity<T, Dimension> const& a,
    quantity<T, Dimension> const& b)
{
  return a.value() != b.value();
}

template <class T, class Dimension>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator>(
    quantity<T, Dimension> const& a,
    quantity<T, Dimension> const& b)
{
  return a.value() > b.value();
}

template <class T, class Dimension>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator>=(
    quantity<T, Dimension> const& a,
    quantity<T, Dimension> const& b)
{
  return a.value() >= b.value();
}

template <class T, class Dimension>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator<=(
    quantity<T, Dimension> const& a,
    quantity<T, Dimension> const& b)
{
  return a.value() <= b.value();
}

template <class T, class Dimension>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator<(
    quantity<T, Dimension> const& a,
    quantity<T, Dimension> const& b)
{
  return a.value() < b.value();
}

template <class T, class Dimension>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline
auto square_root(quantity<T, Dimension> const& a)
{
  using result_dimension = dimension_root<Dimension, 2>;
  return quantity<T, result_dimension>(square_root(a.value()));
}

template <class T, class Dimension>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE
quantity<T, Dimension> absolute_value(quantity<T, Dimension> const& a)
{
  return quantity<T, Dimension>(absolute_value(a.value()));
}

template <class T>
using adimensional_quantity = quantity<T, adimensional>;
template <class T>
using time_quantity = quantity<T, time_dimension>;
template <class T>
using length_quantity = quantity<T, length_dimension>;
template <class T>
using position_quantity = vector3<length_quantity<T>>;
template <class T>
using area_quantity = quantity<T, area_dimension>;
template <class T>
using volume_quantity = quantity<T, volume_dimension>;
template <class T>
using mass_quantity = quantity<T, mass_dimension>;
template <class T>
using temperature_quantity = quantity<T, temperature_dimension>;
template <class T>
using density_quantity = quantity<T, density_dimension>;
template <class T>
using mass_density_quantity = quantity<T, mass_density_dimension>;
template <class T>
using pressure_quantity = quantity<T, pressure_dimension>;
template <class T>
using energy_quantity = quantity<T, energy_dimension>;
template <class T>
using specific_energy_quantity = quantity<T, specific_energy_dimension>;
template <class T>
using specific_energy_rate_quantity = quantity<T, specific_energy_rate_dimension>;
template <class T>
using energy_density_quantity = quantity<T, energy_density_dimension>;
template <class T>
using speed_quantity = quantity<T, velocity_dimension>;
template <class T>
using velocity_quantity = vector3<quantity<T, velocity_dimension>>;
template <class T>
using axial_momentum_quantity = quantity<T, momentum_dimension>;
template <class T>
using axial_momentum_density_quantity = quantity<T, momentum_density_dimension>;
template <class T>
using momentum_quantity = vector3<quantity<T, momentum_dimension>>;
template <class T>
using axial_acceleration_quantity = quantity<T, acceleration_dimension>;
template <class T>
using acceleration_quantity = vector3<axial_acceleration_quantity<T>>;
template <class T>
using axial_force_quantity = quantity<T, force_dimension>;
template <class T>
using force_quantity = vector3<axial_force_quantity<T>>;
template <class T>
using axial_gradient_quantity = quantity<T, gradient_dimension>;
template <class T>
using gradient_quantity = vector3<axial_gradient_quantity<T>>;
template <class T>
using rate_quantity = quantity<T, rate_dimension>;
template <class T>
using velocity_gradient_quantity = matrix3x3<rate_quantity<T>>;
template <class T>
using strain_rate_quantity = symmetric3x3<rate_quantity<T>>;
template <class T>
using kinematic_viscosity_quantity = quantity<T, kinematic_viscosity_dimension>;
template <class T>
using dynamic_viscosity_quantity = quantity<T, dynamic_viscosity_dimension>;
template <class T>
using specific_heat_quantity = quantity<T, specific_heat_dimension>;

template <class T, class Dimension, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE
quantity<simd<T, Abi>, Dimension> load_scalar(
    quantity<T, Dimension> const* ptr, int offset, simd_mask<T, Abi> const& mask)
{
  return simd<T, Abi>::masked_load(&(ptr->value()) + index, mask);
}

template <class T, class Dimension, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE
vector3<quantity<simd<T, Abi>, Dimension>> load_vector3(
    quantity<T, Dimension> const* ptr, int stride, int offset, simd_mask<T, Abi> const& mask)
{
  return vector3<quantity<simd<T, Abi>, Dimension>>(
      simd<T, Abi>::masked_load(&(ptr->value()) + stride * 0 + index, mask),
      simd<T, Abi>::masked_load(&(ptr->value()) + stride * 1 + index, mask),
      simd<T, Abi>::masked_load(&(ptr->value()) + stride * 2 + index, mask));
}

template <class T, class Dimension, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE
symmetric3x3<quantity<simd<T, Abi>, Dimension>> load_symmetric3x3(
    quantity<T, Dimension> const* ptr, int stride, int offset, simd_mask<T, Abi> const& mask)
{
  return symmetric3x3<quantity<simd<T, Abi>, Dimension>>(
      simd<T, Abi>::masked_load(&(ptr->value()) + stride * 0 + index, mask),
      simd<T, Abi>::masked_load(&(ptr->value()) + stride * 1 + index, mask),
      simd<T, Abi>::masked_load(&(ptr->value()) + stride * 2 + index, mask),
      simd<T, Abi>::masked_load(&(ptr->value()) + stride * 3 + index, mask),
      simd<T, Abi>::masked_load(&(ptr->value()) + stride * 4 + index, mask),
      simd<T, Abi>::masked_load(&(ptr->value()) + stride * 5 + index, mask));
}

template <class T, class Dimension, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE
void store(
    quantity<simd<T, Abi>, Dimension> const& q,
    quantity<T, Dimension> const* ptr,
    int offset,
    simd_mask<T, Abi> const& mask)
{
  q.value().masked_store(&(ptr->value()) + index, mask);
}

template <class T, class Dimension, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE
void store(
    vector3<quantity<simd<T, Abi>, Dimension>> const& q,
    quantity<T, Dimension> const* ptr,
    int stride,
    int offset,
    simd_mask<T, Abi> const& mask)
{
  q.x().value().masked_store(&(ptr->value()) + stride * 0 + index, mask);
  q.y().value().masked_store(&(ptr->value()) + stride * 1 + index, mask);
  q.z().value().masked_store(&(ptr->value()) + stride * 2 + index, mask);
}

template <class T, class Dimension, class Abi>
P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE
void store(
    symmetric3x3<quantity<simd<T, Abi>, Dimension>> const& q,
    quantity<T, Dimension> const* ptr,
    int stride,
    int offset,
    simd_mask<T, Abi> const& mask)
{
  q.xx().value().masked_store(&(ptr->value()) + stride * 0 + index, mask);
  q.xy().value().masked_store(&(ptr->value()) + stride * 1 + index, mask);
  q.xz().value().masked_store(&(ptr->value()) + stride * 2 + index, mask);
  q.yy().value().masked_store(&(ptr->value()) + stride * 3 + index, mask);
  q.yz().value().masked_store(&(ptr->value()) + stride * 4 + index, mask);
  q.zz().value().masked_store(&(ptr->value()) + stride * 5 + index, mask);
}

}
