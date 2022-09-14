#pragma once

/* The Kokkos-ified Units Library
 *
 * This is a header-only C++17 library based on Kokkos:
 * https://github.com/kokkos/kokkos
 * This library provides C++ classes to represent physical units
 * and physical quantities (floating-point quantities which have
 * physical units).
 * What distinguishes KUL from similar projects is that it is
 * based on Kokkos so the compile-time-unit quantities can be used
 * on all hardware Kokkos supports including NVIDIA and AMD GPUs
 * inside CUDA and HIP kernels.
 */

#include <string>
#include <memory>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>

namespace kul {

// Section [rational]: constexpr-compatible "runtime" rational number type

KOKKOS_INLINE_FUNCTION constexpr std::int64_t abs(std::int64_t a)
{
  return (a < 0) ? -a : a;
}

KOKKOS_INLINE_FUNCTION constexpr std::int64_t gcd(std::int64_t a, std::int64_t b)
{
  while (b != 0) {
    auto const t = b;
    b = a % b;
    a = t;
  }
  return a;
}

class rational {
  std::int64_t m_numerator{0};
  std::int64_t m_denominator{1};
 public:
  KOKKOS_INLINE_FUNCTION constexpr rational(std::int64_t numerator_arg, std::int64_t denominator_arg)
  {
    auto const abs_num_arg = kul::abs(numerator_arg);
    auto const abs_den_arg = kul::abs(denominator_arg);
    auto const common = kul::gcd(abs_num_arg, abs_den_arg);
    auto const abs_num = abs_num_arg / common;
    auto const abs_den = abs_den_arg / common;
    auto const is_negative = (!(numerator_arg < 0)) != (!(denominator_arg < 0));
    m_numerator = is_negative ? -abs_num : abs_num;
    m_denominator = abs_den;
  }
  KOKKOS_INLINE_FUNCTION constexpr rational(std::int64_t numerator_arg)
    :rational(numerator_arg, 1)
  {
  }
  constexpr rational() = default;
  KOKKOS_INLINE_FUNCTION constexpr std::int64_t numerator() const
  {
    return m_numerator;
  }
  KOKKOS_INLINE_FUNCTION constexpr std::int64_t denominator() const
  {
    return m_denominator;
  }
  template <class T>
  KOKKOS_INLINE_FUNCTION constexpr T convert_to() const
  {
    return T(m_numerator) / T(m_denominator);
  }
};

KOKKOS_INLINE_FUNCTION constexpr rational inverse(rational const& a)
{
  return rational(a.denominator(), a.numerator());
}

KOKKOS_INLINE_FUNCTION constexpr rational operator*(rational const& a, rational const& b)
{
  return rational(a.numerator() * b.numerator(), a.denominator() * b.denominator());
}

KOKKOS_INLINE_FUNCTION constexpr rational operator/(rational const& a, rational const& b)
{
  return a * inverse(b);
}

KOKKOS_INLINE_FUNCTION constexpr bool operator==(rational const& a, rational const& b)
{
  return (a.numerator() == b.numerator()) && (a.denominator() == b.denominator());
}

KOKKOS_INLINE_FUNCTION constexpr bool operator!=(rational const& a, rational const& b)
{
  return !operator==(a, b);
}

KOKKOS_INLINE_FUNCTION constexpr rational pow(rational const& b, int e)
{
  rational result{1};
  for (int i = 0; i < e; ++i) {
    result = result * b;
  }
  for (int i = 0; i < -e; ++i) {
    result = result / b;
  }
  return result;
}

// Section [dimension]: constexpr-compatible "runtime" SI dimension type

class dimension {
  int m_time_exponent;
  int m_length_exponent;
  int m_mass_exponent;
  int m_electric_current_exponent;
  int m_temperature_exponent;
  int m_amount_of_substance_exponent;
  int m_luminous_intensity_exponent;
 public:
  KOKKOS_INLINE_FUNCTION constexpr dimension(
      int time_exponent_arg,
      int length_exponent_arg,
      int mass_exponent_arg,
      int electric_current_exponent_arg = 0,
      int temperature_exponent_arg = 0,
      int amount_of_substance_exponent_arg = 0,
      int luminous_intensity_arg = 0)
    :m_time_exponent(time_exponent_arg)
    ,m_length_exponent(length_exponent_arg)
    ,m_mass_exponent(mass_exponent_arg)
    ,m_electric_current_exponent(electric_current_exponent_arg)
    ,m_temperature_exponent(temperature_exponent_arg)
    ,m_amount_of_substance_exponent(amount_of_substance_exponent_arg)
    ,m_luminous_intensity_exponent(luminous_intensity_arg)
  {
  }
  KOKKOS_INLINE_FUNCTION constexpr int time_exponent() const
  {
    return m_time_exponent;
  }
  KOKKOS_INLINE_FUNCTION constexpr int length_exponent() const
  {
    return m_length_exponent;
  }
  KOKKOS_INLINE_FUNCTION constexpr int mass_exponent() const
  {
    return m_mass_exponent;
  }
  KOKKOS_INLINE_FUNCTION constexpr int electric_current_exponent() const
  {
    return m_electric_current_exponent;
  }
  KOKKOS_INLINE_FUNCTION constexpr int temperature_exponent() const
  {
    return m_temperature_exponent;
  }
  KOKKOS_INLINE_FUNCTION constexpr int amount_of_substance_exponent() const
  {
    return m_amount_of_substance_exponent;
  }
  KOKKOS_INLINE_FUNCTION constexpr int luminous_intensity_exponent() const
  {
    return m_luminous_intensity_exponent;
  }
};

KOKKOS_INLINE_FUNCTION constexpr dimension dimension_one()
{
  return dimension(0, 0, 0);
}


KOKKOS_INLINE_FUNCTION constexpr dimension operator*(dimension const& a, dimension const& b)
{
  return dimension(
      a.time_exponent() + b.time_exponent(),
      a.length_exponent() + b.length_exponent(),
      a.mass_exponent() + b.mass_exponent(),
      a.electric_current_exponent() + b.electric_current_exponent(),
      a.temperature_exponent() + b.temperature_exponent(),
      a.amount_of_substance_exponent() + b.amount_of_substance_exponent(),
      a.luminous_intensity_exponent() + b.luminous_intensity_exponent());
}

KOKKOS_INLINE_FUNCTION constexpr dimension operator/(dimension const& a, dimension const& b)
{
  return dimension(
      a.time_exponent() - b.time_exponent(),
      a.length_exponent() - b.length_exponent(),
      a.mass_exponent() - b.mass_exponent(),
      a.electric_current_exponent() - b.electric_current_exponent(),
      a.temperature_exponent() - b.temperature_exponent(),
      a.amount_of_substance_exponent() - b.amount_of_substance_exponent(),
      a.luminous_intensity_exponent() - b.luminous_intensity_exponent());
}

KOKKOS_INLINE_FUNCTION constexpr dimension pow(dimension const& d, int e)
{
  return dimension(
      d.time_exponent() * e,
      d.length_exponent() * e,
      d.mass_exponent() * e,
      d.electric_current_exponent() * e,
      d.temperature_exponent() * e,
      d.amount_of_substance_exponent() * e,
      d.luminous_intensity_exponent() * e);
}

KOKKOS_INLINE_FUNCTION constexpr bool operator==(dimension const& a, dimension const& b)
{
  return (a.time_exponent() == b.time_exponent()) &&
         (a.length_exponent() == b.length_exponent()) &&
         (a.mass_exponent() == b.mass_exponent()) &&
         (a.electric_current_exponent() == b.electric_current_exponent()) &&
         (a.temperature_exponent() == b.temperature_exponent()) &&
         (a.amount_of_substance_exponent() == b.amount_of_substance_exponent()) &&
         (a.luminous_intensity_exponent() == b.luminous_intensity_exponent());
}

KOKKOS_INLINE_FUNCTION constexpr bool operator!=(dimension const& a, dimension const& b)
{
  return !operator==(a, b);
}

// Section [named dimension]: commonly referred-to dimensions

KOKKOS_INLINE_FUNCTION constexpr dimension time()
{
  return dimension(1, 0, 0);
}

KOKKOS_INLINE_FUNCTION constexpr dimension length()
{
  return dimension(0, 1, 0);
}

KOKKOS_INLINE_FUNCTION constexpr dimension mass()
{
  return dimension(0, 0, 1);
}

KOKKOS_INLINE_FUNCTION constexpr dimension electric_current()
{
  return dimension(0, 0, 0, 1);
}

KOKKOS_INLINE_FUNCTION constexpr dimension temperature()
{
  return dimension(0, 0, 0, 0, 1);
}

KOKKOS_INLINE_FUNCTION constexpr dimension amount_of_substance()
{
  return dimension(0, 0, 0, 0, 0, 1);
}

KOKKOS_INLINE_FUNCTION constexpr dimension luminous_intensity()
{
  return dimension(0, 0, 0, 0, 0, 0, 1);
}

KOKKOS_INLINE_FUNCTION constexpr dimension area()
{
  return length() * length();
}

KOKKOS_INLINE_FUNCTION constexpr dimension volume()
{
  return area() * length();
}

KOKKOS_INLINE_FUNCTION constexpr dimension speed()
{
  return length() / time();
}

KOKKOS_INLINE_FUNCTION constexpr dimension acceleration()
{
  return speed() / time();
}

KOKKOS_INLINE_FUNCTION constexpr dimension force()
{
  return mass() * acceleration();
}

KOKKOS_INLINE_FUNCTION constexpr dimension momentum()
{
  return mass() * speed();
}

KOKKOS_INLINE_FUNCTION constexpr dimension energy()
{
  return force() * length();
}

KOKKOS_INLINE_FUNCTION constexpr dimension pressure()
{
  return force() / area();
}

KOKKOS_INLINE_FUNCTION constexpr dimension electric_charge()
{
  return electric_current() * time();
}

KOKKOS_INLINE_FUNCTION constexpr dimension electric_potential()
{
  return energy() / electric_charge();
}

KOKKOS_INLINE_FUNCTION constexpr dimension electrical_resistance()
{
  return electric_potential() / electric_current();
}

KOKKOS_INLINE_FUNCTION constexpr dimension electrical_conductance()
{
  return dimension_one() / electrical_resistance();
}

KOKKOS_INLINE_FUNCTION constexpr dimension electrical_resistivity()
{
  return electrical_resistance() * length();
}

KOKKOS_INLINE_FUNCTION constexpr dimension electrical_conductivity()
{
  return dimension_one() / electrical_resistivity();
}

KOKKOS_INLINE_FUNCTION constexpr dimension capacitance()
{
  return electric_charge() / electric_potential();
}

KOKKOS_INLINE_FUNCTION constexpr dimension inductance()
{
  return electric_potential() / (electric_current() / time());
}

// Section [optiona]: constexpr-compatible and Kokkosified version of std::optional<T>

class nullopt_t {};
inline constexpr nullopt_t nullopt = {};

// This class always has the value alive because things like placement new don't work
// in a constexpr context, so it is limited in usefulness to trivial types

template <class T>
class optional
{
  bool m_has_value{false};
  T m_value;
 public:
  KOKKOS_INLINE_FUNCTION constexpr optional(nullopt_t)
  {
  }
  KOKKOS_INLINE_FUNCTION constexpr optional(T const& value)
  {
    m_value = T(value);
    m_has_value = true;
  }
  KOKKOS_INLINE_FUNCTION constexpr bool has_value() const
  {
    return m_has_value;
  }
  KOKKOS_INLINE_FUNCTION constexpr T& value()
  {
    return m_value;
  }
  KOKKOS_INLINE_FUNCTION constexpr T const& value() const
  {
    return m_value;
  }
};

template <class T>
KOKKOS_INLINE_FUNCTION constexpr bool operator==(optional<T> const& a, optional<T> const& b)
{
  if ((!a.has_value()) && (!b.has_value())) {
    return true;
  }
  if (a.has_value() && b.has_value()) {
    return a.value() == b.value();
  }
  return false;
}

// Section [unit]: virtual base, derived Curiously Recurring Template Pattern classes for physical unit types,

class unit {
 public:
  virtual ~unit() = default;
  virtual std::string name() const = 0;
  virtual kul::dimension dimension() const = 0; 
  virtual rational magnitude() const = 0;
  virtual optional<rational> origin() const = 0;
  virtual std::unique_ptr<unit> copy() const = 0;
  virtual std::unique_ptr<unit> simplify() const = 0;
};

inline bool operator==(unit const& a, unit const& b)
{
  return a.dimension() == b.dimension() &&
    a.magnitude() == b.magnitude() &&
    a.origin() == b.origin();
}

inline bool operator!=(unit const& a, unit const& b)
{
  return !operator==(a, b);
}

class named : public unit {
 public:
};

template <class Unit>
class crtp : public named {
 public:
  std::string name() const override
  {
    return Unit::static_name();
  }
  kul::dimension dimension() const override 
  {
    return Unit::static_dimension();
  }
  rational magnitude() const override 
  {
    return Unit::static_magnitude();
  }
  optional<rational> origin() const override 
  {
    return Unit::static_origin();
  }
  std::unique_ptr<unit> copy() const override
  {
    return std::make_unique<Unit>();
  }
  std::unique_ptr<unit> simplify() const override
  {
    return this->copy();
  }
};

class unit_one : public crtp<unit_one> {
 public:
  static std::string static_name() { return "1"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return kul::dimension_one(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(1); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

// Section [dynamic]: classes for runtime representation of derived units

class dynamic_unit : public unit {
  std::unique_ptr<unit> m_pointer;
 public:
  dynamic_unit() = default;
  dynamic_unit(unit const& u)
    :m_pointer(u.copy())
  {
  }
  dynamic_unit(std::unique_ptr<unit>&& ptr)
    :m_pointer(std::move(ptr))
  {
  }
  dynamic_unit(std::unique_ptr<unit> const& ptr)
    :m_pointer(ptr->copy())
  {
  }
  dynamic_unit(dynamic_unit&&) = default;
  dynamic_unit& operator=(dynamic_unit&&) = default;
  dynamic_unit(dynamic_unit const& other)
    :m_pointer(other ? other.m_pointer->copy() : std::unique_ptr<unit>())
  {
  }
  dynamic_unit& operator=(dynamic_unit const& other)
  {
    m_pointer = other.m_pointer->copy();
    return *this;
  }
  std::string name() const override
  {
    return m_pointer->name();
  }
  kul::dimension dimension() const override
  {
    return m_pointer->dimension();
  }
  rational magnitude() const override
  {
    return m_pointer->magnitude();
  }
  optional<rational> origin() const override
  {
    return m_pointer->origin();
  }
  std::unique_ptr<unit> copy() const override
  {
    return m_pointer->copy();
  }
  std::unique_ptr<unit> simplify() const override
  {
    return m_pointer->simplify();
  }
  unit const* pointer() const
  {
    return m_pointer.get();
  }
  unit* pointer()
  {
    return m_pointer.get();
  }
  bool is_unitless() const
  {
    return dynamic_cast<unit_one const*>(m_pointer.get()) != nullptr;
  }
  explicit operator bool() const
  {
    return static_cast<bool>(m_pointer);
  }
};

class dynamic_exp : public unit {
  dynamic_unit m_base;
  int m_exponent;
 public:
  dynamic_exp(dynamic_unit base_arg, int exponent_arg)
    :m_base(base_arg)
    ,m_exponent(exponent_arg)
  {
  }
  std::string name() const override
  {
    return m_base.name() + "^" + std::to_string(m_exponent);
  }
  kul::dimension dimension() const override
  {
    return kul::pow(m_base.dimension(), m_exponent);
  }
  rational magnitude() const override
  {
    return kul::pow(m_base.magnitude(), m_exponent);
  }
  optional<rational> origin() const override
  {
    return nullopt;
  }
  std::unique_ptr<unit> copy() const override
  {
    return std::make_unique<dynamic_exp>(*this);
  }
  std::unique_ptr<unit> simplify() const override
  {
    if (m_exponent == 0) return unit_one().copy();
    if (m_exponent == 1) return m_base.copy();
    return copy();
  }
  dynamic_unit const& base() const
  {
    return m_base;
  }
  int exponent() const
  {
    return m_exponent;
  }
};

class dynamic_product : public unit {
  std::vector<dynamic_unit> m_terms;
 public:
  void push_back(dynamic_unit const& term)
  {
    m_terms.push_back(term);
  }
  void push_back_unless_unitless(dynamic_unit const& term)
  {
    if (dynamic_cast<unit_one const*>(term.pointer()) == nullptr) {
      push_back(term);
    }
  }
  void multiply_with(dynamic_exp const& new_exp)
  {
    for (auto& existing_any : m_terms) {
      auto& existing_exp = dynamic_cast<dynamic_exp&>(*(existing_any.pointer()));
      if (existing_exp.base() == new_exp.base()) {
        existing_exp = dynamic_exp(new_exp.base(),
            existing_exp.exponent() + new_exp.exponent());
        return;
      }
    }
    push_back(new_exp);
  }
  void divide_by(dynamic_exp const& new_exp)
  {
    multiply_with(dynamic_exp(new_exp.base(), -new_exp.exponent()));
  }
  void multiply_with(named const& new_named)
  {
    multiply_with(dynamic_exp(new_named, 1));
  }
  void divide_by(named const& new_named)
  {
    divide_by(dynamic_exp(new_named, 1));
  }
  void multiply_with(dynamic_product const& other_product)
  {
    for (auto& term : other_product.m_terms) {
      multiply_with(term);
    }
  }
  void divide_by(dynamic_product const& other_product)
  {
    for (auto& term : other_product.m_terms) {
      divide_by(term);
    }
  }
  void multiply_with(dynamic_unit const& new_unit)
  {
    auto ptr = new_unit.pointer();
    auto product_ptr = dynamic_cast<dynamic_product const*>(ptr);
    if (product_ptr) {
      multiply_with(*product_ptr);
      return;
    }
    auto exp_ptr = dynamic_cast<dynamic_exp const*>(ptr);
    if (exp_ptr) {
      multiply_with(*exp_ptr);
      return;
    }
    auto named_ptr = dynamic_cast<named const*>(ptr);
    if (named_ptr) {
      multiply_with(*named_ptr);
      return;
    }
  }
  void divide_by(dynamic_unit const& new_unit)
  {
    auto ptr = new_unit.pointer();
    auto product_ptr = dynamic_cast<dynamic_product const*>(ptr);
    if (product_ptr) {
      divide_by(*product_ptr);
      return;
    }
    auto exp_ptr = dynamic_cast<dynamic_exp const*>(ptr);
    if (exp_ptr) {
      divide_by(*exp_ptr);
      return;
    }
    auto named_ptr = dynamic_cast<named const*>(ptr);
    if (named_ptr) {
      divide_by(*named_ptr);
      return;
    }
  }
  std::string name() const override
  {
    auto it = m_terms.begin();
    auto result = it->name();
    ++it;
    while (it != m_terms.end()) {
      result += " * ";
      result += it->name();
      ++it;
    }
    return result;
  }
  kul::dimension dimension() const override
  {
    auto it = m_terms.begin();
    auto result = it->dimension();
    ++it;
    while (it != m_terms.end()) {
      result = result * it->dimension();
      ++it;
    }
    return result;
  }
  rational magnitude() const override
  {
    auto it = m_terms.begin();
    auto result = it->magnitude();
    ++it;
    while (it != m_terms.end()) {
      result = result * it->magnitude();
      ++it;
    }
    return result;
  }
  optional<rational> origin() const override
  {
    return nullopt;
  }
  std::unique_ptr<unit> copy() const override
  {
    return std::make_unique<dynamic_product>(*this);
  }
  std::unique_ptr<unit> simplify() const override
  {
    auto result = dynamic_product();
    for (auto& u : m_terms) {
      result.push_back_unless_unitless(u.simplify());
    }
    if (result.m_terms.empty()) return unit_one().copy();
    if (result.m_terms.size() == 1) return result.m_terms.front().copy();
    return result.copy();
  }
  std::vector<dynamic_unit> const& terms() const
  {
    return m_terms;
  }
};

inline dynamic_unit operator*(dynamic_unit const& a, dynamic_unit const& b)
{
  dynamic_product p;
  p.multiply_with(a);
  p.multiply_with(b);
  return p.simplify();
}

inline dynamic_unit operator/(dynamic_unit const& a, dynamic_unit const& b)
{
  dynamic_product p;
  p.multiply_with(a);
  p.divide_by(b);
  return p.simplify();
}

inline dynamic_unit& operator*=(dynamic_unit& a, dynamic_unit const& b)
{
  return a = a * b;
}

inline dynamic_unit& operator/=(dynamic_unit& a, dynamic_unit const& b)
{
  return a = a / b;
}

inline dynamic_unit root(dynamic_unit const& base, int exponent)
{
  auto ptr = base.pointer();
  auto named_ptr = dynamic_cast<named const*>(ptr);
  if (named_ptr) {
    throw std::runtime_error("cannot take " + std::to_string(exponent) + "th root of named unit");
  }
  auto exp_ptr = dynamic_cast<dynamic_exp const*>(ptr);
  if (exp_ptr) {
    if (exp_ptr->exponent() % exponent != 0) {
      throw std::runtime_error("taking " + std::to_string(exponent) + "th root of non-divisible "
          + std::to_string(exp_ptr->exponent()) + "th power of " + exp_ptr->base().name());
    }
    return dynamic_exp(exp_ptr->base(), exp_ptr->exponent() / exponent).simplify();
  }
  auto product_ptr = dynamic_cast<dynamic_product const*>(ptr);
  if (product_ptr) {
    auto result = dynamic_product();
    for (auto& term : product_ptr->terms()) {
      result.multiply_with(root(term, exponent));
    }
    return result.simplify();
  }
  throw std::logic_error("unexpected type");
}

inline dynamic_unit sqrt(dynamic_unit const& base)
{
  return root(base, 2);
}

inline dynamic_unit cbrt(dynamic_unit const& base)
{
  return root(base, 3);
}

// Section [static]: Compile-time implementations of derived unit operations

template <class Base, int Exponent>
class static_pow : public crtp<static_pow<Base, Exponent>> {
 public:
  static std::string static_name() { return Base::static_name() + "^" + std::to_string(Exponent); }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return kul::pow(Base::static_dimension(), Exponent); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return kul::pow(Base::static_magnitude(), Exponent); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
  std::unique_ptr<unit> copy() const override
  {
    return dynamic_exp(Base(), Exponent).copy();
  }
};

template <class... Units>
class static_product;

template <>
class static_product<> : public crtp<static_product<>> {
 public:
  static std::string static_name() { return "1"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return kul::dimension_one(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return kul::rational(1); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
  std::unique_ptr<unit> copy() const override
  {
    return dynamic_product().copy();
  }
};

template <class Unit>
class static_product<Unit> : public crtp<static_product<Unit>> {
 public:
  static std::string static_name() { return Unit::static_name(); }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return Unit::static_dimension(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return Unit::static_magnitude(); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
  std::unique_ptr<unit> copy() const override
  {
    auto p = dynamic_product();
    p.multiply_with(Unit());
    return p.copy();
  }
};

template <class FirstUnit, class... OtherUnits>
class static_product<FirstUnit, OtherUnits...> : public crtp<static_product<FirstUnit, OtherUnits...>> {
 public:
  using tail_type = static_product<OtherUnits...>;
  static std::string static_name()
  {
    return FirstUnit::static_name() + " * " + tail_type::static_name();
  }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension()
  {
    return FirstUnit::static_dimension() * tail_type::static_dimension();
  }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude()
  {
    return FirstUnit::static_magnitude() * tail_type::static_magnitude();
  }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
  std::unique_ptr<unit> copy() const override
  {
    auto p = dynamic_product();
    p.multiply_with(FirstUnit());
    p.multiply_with(tail_type());
    return p.copy();
  }
};

template <class A, class B>
class push_back;

template <class... Units, class LastUnit>
class push_back<static_product<Units...>, LastUnit> {
 public:
  using type = static_product<Units..., LastUnit>;
};

template <class A, class B>
using push_back_t = typename push_back<A, B>::type;

template <class A, class B>
class prepend;

template <class FirstUnit, class... Units>
class prepend<FirstUnit, static_product<Units...>> {
 public:
  using type = static_product<FirstUnit, Units...>;
};

template <class A, class B>
using prepend_t = typename prepend<A, B>::type;

template <class A, class B>
class push_back_unless_unitless {
 public:
  using type = push_back_t<A, B>;
};

template <class A>
class push_back_unless_unitless<A, unit_one> {
 public:
  using type = A;
};

template <class A, class B>
using push_back_unless_unitless_t = typename push_back_unless_unitless<A, B>::type;

template <class A, class B>
class prepend_unless_unitless {
 public:
  using type = prepend_t<A, B>;
};

template <class B>
class prepend_unless_unitless<unit_one, B> {
 public:
  using type = B;
};

template <class A, class B>
using prepend_unless_unitless_t = typename prepend_unless_unitless<A, B>::type;

template <class A, class B>
class multiply_with {
 public:
  using type = typename multiply_with<A, static_pow<B, 1>>::type;
};

template <class Base, int Exponent>
class multiply_with<static_product<>, static_pow<Base, Exponent>> {
 public:
  using type = static_product<static_pow<Base, Exponent>>;
};

template <class FirstUnit, class... NextUnits, class Base, int Exponent>
class multiply_with<static_product<FirstUnit, NextUnits...>, static_pow<Base, Exponent>> {
 public:
  using type = prepend_t<FirstUnit, typename multiply_with<static_product<NextUnits...>, static_pow<Base, Exponent>>::type>;
};

template <class... NextUnits, class Base, int Exponent1, int Exponent2>
class multiply_with<static_product<static_pow<Base, Exponent1>, NextUnits...>, static_pow<Base, Exponent2>> {
 public:
  using type = prepend_t<static_pow<Base, Exponent1 + Exponent2>, static_product<NextUnits...>>;
};

template <class LHS>
class multiply_with<LHS, static_product<>> {
 public:
  using type = LHS;
};

template <class LHS, class FirstUnit, class... NextUnits>
class multiply_with<LHS, static_product<FirstUnit, NextUnits...>> {
  using product_with_first = typename multiply_with<LHS, FirstUnit>::type;
 public:
  using type = typename multiply_with<product_with_first, static_product<NextUnits...>>::type;
};

template <class A, class B>
using multiply_with_t = typename multiply_with<A, B>::type;

template <class A, class B>
class divide_by {
 public:
  using type = typename divide_by<A, static_pow<B, 1>>::type;
};

template <class Base, int Exponent>
class divide_by<static_product<>, static_pow<Base, Exponent>> {
 public:
  using type = static_product<static_pow<Base, -Exponent>>;
};

template <class FirstUnit, class... NextUnits, class Base, int Exponent>
class divide_by<static_product<FirstUnit, NextUnits...>, static_pow<Base, Exponent>> {
 public:
  using type = prepend_t<FirstUnit, typename divide_by<static_product<NextUnits...>, static_pow<Base, Exponent>>::type>;
};

template <class... NextUnits, class Base, int Exponent1, int Exponent2>
class divide_by<static_product<static_pow<Base, Exponent1>, NextUnits...>, static_pow<Base, Exponent2>> {
 public:
  using type = prepend_t<static_pow<Base, Exponent1 - Exponent2>, static_product<NextUnits...>>;
};

template <class LHS>
class divide_by<LHS, static_product<>> {
 public:
  using type = LHS;
};

template <class LHS, class FirstUnit, class... NextUnits>
class divide_by<LHS, static_product<FirstUnit, NextUnits...>> {
  using product_with_first = typename divide_by<LHS, FirstUnit>::type;
 public:
  using type = typename divide_by<product_with_first, static_product<NextUnits...>>::type;
};

template <class A, class B>
using divide_by_t = typename divide_by<A, B>::type;

template <class T>
class simplify {
 public:
  using type = T;
};

template <class Base>
class simplify<static_pow<Base, 0>> {
 public:
  using type = unit_one;
};

template <class Base>
class simplify<static_pow<Base, 1>> {
 public:
  using type = Base;
};

template <int Exponent>
class simplify<static_pow<unit_one, Exponent>> {
 public:
  using type = unit_one;
};

template <>
class simplify<static_pow<unit_one, 0>> {
 public:
  using type = unit_one;
};

template <>
class simplify<static_pow<unit_one, 1>> {
 public:
  using type = unit_one;
};

template <class T>
class simplify_terms;

template <>
class simplify_terms<static_product<>> {
 public:
  using type = static_product<>;
};

template <class FirstUnit, class... NextUnits>
class simplify_terms<static_product<FirstUnit, NextUnits...>> {
  using simplified_first_unit = typename simplify<FirstUnit>::type;
  using simplified_next_units = typename simplify_terms<static_product<NextUnits...>>::type;
 public:
  using type = prepend_unless_unitless_t<simplified_first_unit, simplified_next_units>;
};

template <class T>
class simplify_product {
 public:
  using type = T;
};

template <>
class simplify_product<static_product<>> {
 public:
  using type = unit_one;
};

template <class T>
class simplify_product<static_product<T>> {
 public:
  using type = T;
};

template <class... Terms>
class simplify<static_product<Terms...>> {
  using simplified_terms = typename simplify_terms<static_product<Terms...>>::type;
 public:
  using type = typename simplify_product<simplified_terms>::type;
};

template <class T>
using simplify_t = typename simplify<T>::type;

template <class T>
using canonicalize_t = multiply_with_t<static_product<>, T>;

template <class A, class B>
using multiply = simplify_t<multiply_with_t<canonicalize_t<A>, B>>;

template <class A, class B>
using divide = simplify_t<divide_by_t<canonicalize_t<A>, B>>;

template <class T>
using reciprocal = divide<unit_one, T>;

template <class A, int Exponent>
class static_root_helper;

template <class A, int Exponent1, int Exponent2>
class static_root_helper<static_pow<A, Exponent1>, Exponent2> {
 public:
  static_assert(Exponent1 % Exponent2 == 0, "taking root of non-divisible power");
  using type = static_pow<A, Exponent1 / Exponent2>;
};

template <int Exponent1, int Exponent2>
class static_root_helper<static_pow<unit_one, Exponent1>, Exponent2> {
 public:
  using type = static_pow<unit_one, 1>;
};

template <int Exponent>
class static_root_helper<static_product<>, Exponent> {
 public:
  using type = static_product<>;
};

template <int Exponent, class FirstUnit, class... NextUnits>
class static_root_helper<static_product<FirstUnit, NextUnits...>, Exponent> {
  using first_root = typename static_root_helper<FirstUnit, Exponent>::type;
  using next_root = typename static_root_helper<static_product<NextUnits...>, Exponent>::type;
 public:
  using type = prepend_t<first_root, next_root>;
};

template <class Base, int Exponent>
using static_root = simplify_t<typename static_root_helper<canonicalize_t<Base>, Exponent>::type>;

template <class Base>
using static_sqrt = static_root<Base, 2>;

template <class Base>
using static_cbrt = static_root<Base, 3>;

template <class A, class B>
inline constexpr bool are_equal =
    A::static_dimension() == B::static_dimension() &&
    A::static_magnitude() == B::static_magnitude() &&
    A::static_origin() == B::static_origin();

template <class T>
class make_relative : public crtp<make_relative<T>> {
 public:
  static std::string static_name() { return T::static_name(); }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return T::static_dimension(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return T::static_magnitude(); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

template <class T>
inline constexpr bool is_absolute = T::static_origin().has_value();

template <class T>
inline constexpr bool is_relative = !is_absolute<T>;

// Section [convert]: a constexpr-compatible class representing the conversion between two units

template <class T>
class conversion {
  T m_multiplier;
  T m_offset;
 public:
  KOKKOS_INLINE_FUNCTION constexpr conversion(
      rational const& old_magnitude,
      optional<rational> const& old_origin,
      rational const& new_magnitude,
      optional<rational> const& new_origin)
    :m_multiplier((old_magnitude / new_magnitude).convert_to<T>())
    ,m_offset(0)
  {
    if (old_origin.has_value()) m_offset += (old_origin.value() / new_magnitude).convert_to<T>();
    if (new_origin.has_value()) m_offset -= (new_origin.value() / new_magnitude).convert_to<T>();
  }
  inline conversion(unit const& from, unit const& to)
    :conversion(
        from.magnitude(),
        from.origin(),
        to.magnitude(),
        to.origin())
  {
  }
  KOKKOS_INLINE_FUNCTION constexpr T operator()(T const& old_value) const
  {
    return old_value * m_multiplier + m_offset;
  }
};

template <class T, class From, class To>
inline constexpr conversion<T> static_conversion = conversion<T>(
    From::static_magnitude(), From::static_origin(),
    To::static_magnitude(), To::static_origin());

// Section [prefix]: class template versions of metric prefixes

template <class T>
class giga : public crtp<giga<T>> {
 public:
  static std::string static_name() { return "G" + T::static_name(); }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return T::static_dimension(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(1'000'000'000) * T::static_magnitude(); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

template <class T>
class mega : public crtp<mega<T>> {
 public:
  static std::string static_name() { return "M" + T::static_name(); }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return T::static_dimension(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(1'000'000) * T::static_magnitude(); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

template <class T>
class kilo : public crtp<kilo<T>> {
 public:
  static std::string static_name() { return "k" + T::static_name(); }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return T::static_dimension(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(1000) * T::static_magnitude(); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

template <class T>
class centi : public crtp<centi<T>> {
 public:
  static std::string static_name() { return "c" + T::static_name(); }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return T::static_dimension(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(1, 100) * T::static_magnitude(); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

template <class T>
class milli : public crtp<milli<T>> {
 public:
  static std::string static_name() { return "m" + T::static_name(); }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return T::static_dimension(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(1, 1000) * T::static_magnitude(); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

// Section [named]: units accepted in physics with their own symbol

class second : public crtp<second> {
 public:
  static std::string static_name() { return "s"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return time(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(1); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

class meter : public crtp<meter> {
 public:
  static std::string static_name() { return "m"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return length(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(1); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

using centimeter = centi<meter>;

class inch : public crtp<inch> {
 public:
  static std::string static_name() { return "in"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return length(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(254, 10'000); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

class gram : public crtp<gram> {
 public:
  static std::string static_name() { return "g"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return mass(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(1, 1000); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

class radian : public crtp<radian> {
 public:
  static std::string static_name() { return "rad"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return dimension_one(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(1); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

class kelvin : public crtp<kelvin> {
 public:
  static std::string static_name() { return "K"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return temperature(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(1); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return rational(0); }
};

class mole : public crtp<mole> {
 public:
  static std::string static_name() { return "mol"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return amount_of_substance(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(1); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

class candela : public crtp<candela> {
 public:
  static std::string static_name() { return "cd"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return luminous_intensity(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(1); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

class temperature_electronvolt : public crtp<temperature_electronvolt> {
 public:
  static std::string static_name() { return "eV/k_B"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return temperature(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() {
    return rational(11'604'518'12, 1'000'00); // 11 604 . 518 12
  }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return rational(0); }
};

class ampere : public crtp<ampere> {
 public:
  static std::string static_name() { return "A"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return electric_current(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(1); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

class coulomb : public crtp<coulomb> {
 public:
  static std::string static_name() { return "C"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return electric_charge(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(1); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

// the unit of current in the electrostatic centimeter-gram-second unit system, ESU, a.k.a Gaussian units

class statampere : public crtp<statampere> {
 public:
  static std::string static_name() { return "statA"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return electric_current(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() {
    auto constexpr speed_of_light_in_centimeters_per_second = rational(29979245800);
    return rational(10) / speed_of_light_in_centimeters_per_second;
  }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return rational(0); }
};

class statcoulomb : public crtp<statcoulomb> {
 public:
  static std::string static_name() { return "statC"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return electric_charge(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() {
    return statampere::static_magnitude() * second::static_magnitude();
  }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

class pascal : public crtp<pascal> {
 public:
  static std::string static_name() { return "Pa"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return pressure(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(1); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

class joule : public crtp<joule> {
 public:
  static std::string static_name() { return "J"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return energy(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(1); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

class newton : public crtp<newton> {
 public:
  static std::string static_name() { return "N"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return force(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(1); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

class erg : public crtp<erg> {
 public:
  static std::string static_name() { return "erg"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return energy(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(1, 10'000'000); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

class volt : public crtp<volt> {
 public:
  static std::string static_name() { return "V"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return electric_potential(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(1); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

class statvolt : public crtp<statvolt> {
 public:
  static std::string static_name() { return "statV"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return electric_potential(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() {
    return erg::static_magnitude() / statcoulomb::static_magnitude();
  }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

class ohm : public crtp<ohm> {
 public:
  static std::string static_name() { return "Ohm"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return electrical_resistance(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(1); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

class siemens : public crtp<siemens> {
 public:
  static std::string static_name() { return "S"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return electrical_conductance(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(1); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

class farad : public crtp<farad> {
 public:
  static std::string static_name() { return "F"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return capacitance(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(1); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

class henry : public crtp<henry> {
 public:
  static std::string static_name() { return "H"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return inductance(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() { return rational(1); }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

class gaussian_resistance : public crtp<gaussian_resistance> {
 public:
  static std::string static_name() { return "s/cm"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return electrical_resistance(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() {
    return statvolt::static_magnitude() / statampere::static_magnitude();
  }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

class gaussian_resistivity : public crtp<gaussian_resistivity> {
 public:
  static std::string static_name() { return "s"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return electrical_resistivity(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() {
    return gaussian_resistance::static_magnitude() * centimeter::static_magnitude();
  }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

class gaussian_conductivity : public crtp<gaussian_conductivity> {
 public:
  static std::string static_name() { return "1/s"; }
  KOKKOS_INLINE_FUNCTION static constexpr kul::dimension static_dimension() { return electrical_conductivity(); }
  KOKKOS_INLINE_FUNCTION static constexpr rational static_magnitude() {
    return rational(1) / gaussian_resistivity::static_magnitude();
  }
  KOKKOS_INLINE_FUNCTION static constexpr optional<rational> static_origin() { return nullopt; }
};

using kilogram = kilo<gram>;
using gigapascal = giga<pascal>;
using megajoule = mega<joule>;
using meter_per_second = divide<meter, second>;
using square_meter = multiply<meter, meter>;
using cubic_meter = multiply<square_meter, meter>;
using square_centimeter = multiply<centimeter, centimeter>;
using cubic_centimeter = multiply<square_centimeter, centimeter>;
using kilogram_per_cubic_meter = divide<kilogram, cubic_meter>;
using kilogram_meter_per_second = multiply<kilogram, meter_per_second>;
using gram_per_cubic_centimeter = divide<gram, cubic_centimeter>;
using joule_per_kilogram = divide<joule, kilogram>;
using joule_per_kilogram_per_kelvin = divide<joule_per_kilogram, kelvin>;
using megajoule_per_kilogram = divide<megajoule, kilogram>;
using siemens_per_meter = divide<siemens, meter>;

// Section [quantity]: class template for runtime value with associated unit

template <class T, class Unit = dynamic_unit>
class quantity {
  T m_value;
 public:
  using value_type = T;
  using unit_type = Unit;
  KOKKOS_DEFAULTED_FUNCTION quantity() = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr quantity(quantity&&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr quantity(quantity const&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr quantity& operator=(quantity&&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr quantity& operator=(quantity const&) = default;
  template <class U,
      std::enable_if_t<
        (!std::is_same_v<Unit, unit_one>) && (std::is_constructible_v<value_type, U>),
        bool> = false>
  KOKKOS_INLINE_FUNCTION constexpr explicit quantity(U const& v)
    :m_value(v)
  {
  }
  template <class U,
      std::enable_if_t<
        std::is_same_v<Unit, unit_one> && (std::is_constructible_v<value_type, U>),
        bool> = false>
  KOKKOS_INLINE_FUNCTION constexpr quantity(U const& v)
    :m_value(v)
  {
  }
  KOKKOS_INLINE_FUNCTION constexpr value_type const& value() const { return m_value; }
  KOKKOS_INLINE_FUNCTION constexpr value_type& value() { return m_value; }
  template <class T2, class Unit2,
      std::enable_if_t<are_equal<Unit, Unit2>, bool> = false>
  KOKKOS_INLINE_FUNCTION constexpr
  quantity(quantity<T2, Unit2> const& other)
    :m_value(other.value())
  {
  }
  template <class T2, class Unit2,
      std::enable_if_t<!are_equal<Unit, Unit2>, bool> = false>
  KOKKOS_INLINE_FUNCTION constexpr
  quantity(quantity<T2, Unit2> const& other)
    :m_value(static_conversion<T2, Unit2, Unit>(other.value()))
  {
    static_assert(Unit::static_dimension() == Unit2::static_dimension(),
        "cannot convert between quantities with different dimensions");
    static_assert(
        (is_absolute<Unit> && is_absolute<Unit2>) ||
        (is_relative<Unit> && is_relative<Unit2>),
        "cannot convert from absolute to relative or vice-versa");
  }
  static std::string unit_name() { return unit_type::static_name(); }
  KOKKOS_INLINE_FUNCTION static constexpr
  dimension dimension() { return unit_type::static_dimension(); }
  KOKKOS_INLINE_FUNCTION static constexpr
  rational unit_magnitude() { return unit_type::static_magnitude(); }
  KOKKOS_INLINE_FUNCTION static constexpr
  optional<rational> unit_origin() { return unit_type::static_origin(); }
};

template <class T>
using unitless = quantity<T, unit_one>;

template <class T, class Unit>
KOKKOS_INLINE_FUNCTION constexpr auto operator==(quantity<T, Unit> const& a, quantity<T, Unit> const& b)
{
  return a.value() == b.value();
}

template <class Arithmetic, class T,
  std::enable_if_t<std::is_arithmetic_v<Arithmetic>, bool> = false>
KOKKOS_INLINE_FUNCTION constexpr auto operator==(Arithmetic const& a, quantity<T, unit_one> const& b)
{
  return a == b.value();
}

template <class Arithmetic, class T,
  std::enable_if_t<std::is_arithmetic_v<Arithmetic>, bool> = false>
KOKKOS_INLINE_FUNCTION constexpr auto operator==(quantity<T, unit_one> const& a, Arithmetic const& b)
{
  return a.value() == b;
}

template <class T, class Unit>
KOKKOS_INLINE_FUNCTION constexpr auto operator!=(quantity<T, Unit> const& a, quantity<T, Unit> const& b)
{
  return a.value() != b.value();
}

template <class Arithmetic, class T,
  std::enable_if_t<std::is_arithmetic_v<Arithmetic>, bool> = false>
KOKKOS_INLINE_FUNCTION constexpr auto operator!=(Arithmetic const& a, quantity<T, unit_one> const& b)
{
  return a != b.value();
}

template <class Arithmetic, class T,
  std::enable_if_t<std::is_arithmetic_v<Arithmetic>, bool> = false>
KOKKOS_INLINE_FUNCTION constexpr auto operator!=(quantity<T, unit_one> const& a, Arithmetic const& b)
{
  return a.value() != b;
}

template <class T, class Unit>
KOKKOS_INLINE_FUNCTION constexpr auto operator<=(quantity<T, Unit> const& a, quantity<T, Unit> const& b)
{
  return a.value() <= b.value();
}

template <class Arithmetic, class T,
  std::enable_if_t<std::is_arithmetic_v<Arithmetic>, bool> = false>
KOKKOS_INLINE_FUNCTION constexpr auto operator<=(Arithmetic const& a, quantity<T, unit_one> const& b)
{
  return a <= b.value();
}

template <class Arithmetic, class T,
  std::enable_if_t<std::is_arithmetic_v<Arithmetic>, bool> = false>
KOKKOS_INLINE_FUNCTION constexpr auto operator<=(quantity<T, unit_one> const& a, Arithmetic const& b)
{
  return a.value() <= b;
}

template <class T, class Unit>
KOKKOS_INLINE_FUNCTION constexpr auto operator>=(quantity<T, Unit> const& a, quantity<T, Unit> const& b)
{
  return a.value() >= b.value();
}

template <class Arithmetic, class T,
  std::enable_if_t<std::is_arithmetic_v<Arithmetic>, bool> = false>
KOKKOS_INLINE_FUNCTION constexpr auto operator>=(Arithmetic const& a, quantity<T, unit_one> const& b)
{
  return a >= b.value();
}

template <class Arithmetic, class T,
  std::enable_if_t<std::is_arithmetic_v<Arithmetic>, bool> = false>
KOKKOS_INLINE_FUNCTION constexpr auto operator>=(quantity<T, unit_one> const& a, Arithmetic const& b)
{
  return a.value() >= b;
}

template <class T, class Unit>
KOKKOS_INLINE_FUNCTION constexpr auto operator<(quantity<T, Unit> const& a, quantity<T, Unit> const& b)
{
  return a.value() < b.value();
}

template <class Arithmetic, class T,
  std::enable_if_t<std::is_arithmetic_v<Arithmetic>, bool> = false>
KOKKOS_INLINE_FUNCTION constexpr auto operator>(Arithmetic const& a, quantity<T, unit_one> const& b)
{
  return a > b.value();
}

template <class Arithmetic, class T,
  std::enable_if_t<std::is_arithmetic_v<Arithmetic>, bool> = false>
KOKKOS_INLINE_FUNCTION constexpr auto operator>(quantity<T, unit_one> const& a, Arithmetic const& b)
{
  return a.value() > b;
}

template <class T, class Unit>
KOKKOS_INLINE_FUNCTION constexpr auto operator>(quantity<T, Unit> const& a, quantity<T, Unit> const& b)
{
  return a.value() > b.value();
}

template <class Arithmetic, class T,
  std::enable_if_t<std::is_arithmetic_v<Arithmetic>, bool> = false>
KOKKOS_INLINE_FUNCTION constexpr auto operator<(Arithmetic const& a, quantity<T, unit_one> const& b)
{
  return a < b.value();
}

template <class Arithmetic, class T,
  std::enable_if_t<std::is_arithmetic_v<Arithmetic>, bool> = false>
KOKKOS_INLINE_FUNCTION constexpr auto operator<(quantity<T, unit_one> const& a, Arithmetic const& b)
{
  return a.value() < b;
}

template <class T1, class Unit1, class T2, class Unit2>
KOKKOS_INLINE_FUNCTION constexpr auto operator+(quantity<T1, Unit1> const& a, quantity<T2, Unit2> const& b)
{
  static_assert(Unit1::static_dimension() == Unit2::static_dimension(),
      "cannot add quantities with different physical dimension");
  static_assert(!(is_absolute<Unit1> && is_absolute<Unit2>),
      "cannot add two absolute quantities");
  using value_type = decltype(a.value() + b.value());
  if constexpr (is_relative<Unit1> && is_relative<Unit2>) {
    auto const converted_b = quantity<T2, Unit1>(b);
    return quantity<value_type, Unit1>(a.value() + converted_b.value());
  }
  if constexpr (is_absolute<Unit1> && is_relative<Unit2>) {
    auto const converted_b = quantity<T2, make_relative<Unit1>>(b);
    return quantity<value_type, Unit1>(a.value() + converted_b.value());
  }
  if constexpr (is_relative<Unit1> && is_absolute<Unit2>) {
    return b + a;
  }
}

template <class Arithmetic, class T,
         std::enable_if_t<std::is_arithmetic_v<Arithmetic>, bool> = false>
KOKKOS_INLINE_FUNCTION constexpr auto operator+(Arithmetic const& a, quantity<T, unit_one> const& b)
{
  return quantity<Arithmetic, unit_one>(a) + b;
}

template <class Arithmetic, class T,
         std::enable_if_t<std::is_arithmetic_v<Arithmetic>, bool> = false>
KOKKOS_INLINE_FUNCTION constexpr auto operator+(quantity<T, unit_one> const& a, Arithmetic const& b)
{
  return a + quantity<Arithmetic, unit_one>(b);
}

template <class T1, class Unit1, class T2, class Unit2>
KOKKOS_INLINE_FUNCTION constexpr auto operator-(quantity<T1, Unit1> const& a, quantity<T2, Unit2> const& b)
{
  static_assert(Unit1::static_dimension() == Unit2::static_dimension(),
      "cannot subtract units with different physical dimension");
  static_assert(!(is_relative<Unit1> && is_absolute<Unit2>),
      "cannot subtract an absolute quantity from a relative one");
  using value_type = decltype(a.value() - b.value());
  if constexpr (is_relative<Unit1> && is_relative<Unit2>) {
    auto const converted_b = quantity<T2, Unit1>(b);
    return quantity<value_type, Unit1>(a.value() - converted_b.value());
  }
  if constexpr (is_absolute<Unit1> && is_absolute<Unit2>) {
    auto const converted_b = quantity<T2, Unit1>(b);
    return quantity<value_type, make_relative<Unit1>>(a.value() - converted_b.value());
  }
  if constexpr (is_absolute<Unit1> && is_relative<Unit2>) {
    auto const converted_b = quantity<T2, make_relative<Unit1>>(b);
    return quantity<value_type, Unit1>(a.value() - converted_b.value());
  }
}

template <class T1, class Unit, class T2,
    std::enable_if_t<is_absolute<Unit>, bool> = false>
KOKKOS_INLINE_FUNCTION constexpr auto operator-(quantity<T1, Unit> const& a, quantity<T2, Unit> const& b)
{
  using T3 = decltype(a.value() - b.value());
  return quantity<T3, make_relative<Unit>>(a.value() - b.value());
}

template <class Arithmetic, class T,
         std::enable_if_t<std::is_arithmetic_v<Arithmetic>, bool> = false>
KOKKOS_INLINE_FUNCTION constexpr auto operator-(Arithmetic const& a, quantity<T, unit_one> const& b)
{
  return quantity<Arithmetic, unit_one>(a) - b;
}

template <class Arithmetic, class T,
         std::enable_if_t<std::is_arithmetic_v<Arithmetic>, bool> = false>
KOKKOS_INLINE_FUNCTION constexpr auto operator-(quantity<T, unit_one> const& a, Arithmetic const& b)
{
  return a - quantity<Arithmetic, unit_one>(b);
}

template <class T, class Unit>
KOKKOS_INLINE_FUNCTION constexpr auto operator-(quantity<T, Unit> const& a)
{
  return quantity<T, Unit>(-a.value());
}

template <class T1, class Unit1, class T2, class Unit2>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(quantity<T1, Unit1> const& a, quantity<T2, Unit2> const& b)
{
  using T3 = decltype(a.value() * b.value());
  using Unit3 = multiply<Unit1, Unit2>;
  return quantity<T3, Unit3>(a.value() * b.value());
}

template <class Arithmetic, class T, class Unit,
         std::enable_if_t<std::is_arithmetic_v<Arithmetic>, bool> = false>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(Arithmetic const& a, quantity<T, Unit> const& b)
{
  return quantity<Arithmetic, unit_one>(a) * b;
}

template <class Arithmetic, class T, class Unit,
         std::enable_if_t<std::is_arithmetic_v<Arithmetic>, bool> = false>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(quantity<T, Unit> const& a, Arithmetic const& b)
{
  return a * quantity<Arithmetic, unit_one>(b);
}

template <class T, class Unit, class U>
KOKKOS_INLINE_FUNCTION constexpr quantity<T, Unit>& operator+=(quantity<T, Unit>& a, U const& b)
{
  return a = a + b;
}

template <class T, class Unit, class U>
KOKKOS_INLINE_FUNCTION constexpr quantity<T, Unit>& operator-=(quantity<T, Unit>& a, U const& b)
{
  return a = a - b;
}

template <class T, class Unit, class U>
KOKKOS_INLINE_FUNCTION constexpr quantity<T, Unit>& operator*=(quantity<T, Unit>& a, U const& b)
{
  return a = a * b;
}

template <class T, class Unit, class U>
KOKKOS_INLINE_FUNCTION constexpr quantity<T, Unit>& operator/=(quantity<T, Unit>& a, U const& b)
{
  return a = a / b;
}

// multiplying a floating-point literal by a compile-time unit type

template <class Arithmetic, class Unit,
         std::enable_if_t<std::is_arithmetic_v<Arithmetic>, bool> = false>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(Arithmetic const& a, crtp<Unit> const& b)
{
  return quantity<Arithmetic, Unit>(a);
}

template <class T1, class Unit1, class T2, class Unit2>
KOKKOS_INLINE_FUNCTION constexpr auto operator/(quantity<T1, Unit1> const& a, quantity<T2, Unit2> const& b)
{
  using T3 = decltype(a.value() / b.value());
  using Unit3 = divide<Unit1, Unit2>;
  return quantity<T3, Unit3>(a.value() / b.value());
}

template <class Arithmetic, class T, class Unit,
         std::enable_if_t<std::is_arithmetic_v<Arithmetic>, bool> = false>
KOKKOS_INLINE_FUNCTION constexpr auto operator/(Arithmetic const& a, quantity<T, Unit> const& b)
{
  return quantity<Arithmetic, unit_one>(a) / b;
}

template <class Arithmetic, class T, class Unit,
         std::enable_if_t<std::is_arithmetic_v<Arithmetic>, bool> = false>
KOKKOS_INLINE_FUNCTION constexpr auto operator/(quantity<T, Unit> const& a, Arithmetic const& b)
{
  return a / quantity<Arithmetic, unit_one>(b);
}

template <class T, class Unit>
KOKKOS_INLINE_FUNCTION constexpr auto abs(quantity<T, Unit> const& q)
{
  return quantity<T, Unit>(Kokkos::abs(q.value())); 
}

template <class T, class Unit>
KOKKOS_INLINE_FUNCTION constexpr auto sqrt(quantity<T, Unit> const& q)
{
  return quantity<T, static_sqrt<Unit>>(Kokkos::sqrt(q.value()));
}

template <class T, class Unit>
KOKKOS_INLINE_FUNCTION constexpr auto cbrt(quantity<T, Unit> const& q)
{
  return quantity<T, static_cbrt<Unit>>(Kokkos::cbrt(q.value()));
}

#define KUL_UNITLESS_UNARY_FUNCTION(FUNC) \
template <class T> \
KOKKOS_INLINE_FUNCTION constexpr auto FUNC(quantity<T, unit_one> const& q) \
{ \
  return quantity<T, unit_one>(Kokkos::FUNC(q.value())); \
}

KUL_UNITLESS_UNARY_FUNCTION(exp)
KUL_UNITLESS_UNARY_FUNCTION(exp2)
KUL_UNITLESS_UNARY_FUNCTION(log)
KUL_UNITLESS_UNARY_FUNCTION(log10)
KUL_UNITLESS_UNARY_FUNCTION(log2)
KUL_UNITLESS_UNARY_FUNCTION(erf)
KUL_UNITLESS_UNARY_FUNCTION(erfc)
KUL_UNITLESS_UNARY_FUNCTION(tgamma)
KUL_UNITLESS_UNARY_FUNCTION(lgamma)

#undef KUL_UNITLESS_UNARY_FUNCTION

#define KUL_UNARY_TRIG_FUNCTION(FUNC) \
template <class T> \
KOKKOS_INLINE_FUNCTION constexpr auto FUNC(quantity<T, radian> const& q) \
{ \
  return quantity<T, unit_one>(Kokkos::FUNC(q.value())); \
} \
\
template <class T> \
KOKKOS_INLINE_FUNCTION constexpr auto FUNC(quantity<T, unit_one> const& q) \
{ \
  return FUNC(quantity<T, radian>(q)); \
}

KUL_UNARY_TRIG_FUNCTION(sin)
KUL_UNARY_TRIG_FUNCTION(cos)
KUL_UNARY_TRIG_FUNCTION(tan)
KUL_UNARY_TRIG_FUNCTION(sinh)
KUL_UNARY_TRIG_FUNCTION(cosh)
KUL_UNARY_TRIG_FUNCTION(tanh)

#undef KUL_UNARY_TRIG_FUNCTION

#define KUL_UNARY_INVERSE_TRIG_FUNCTION(FUNC) \
template <class T> \
KOKKOS_INLINE_FUNCTION constexpr auto FUNC(quantity<T, unit_one> const& q) \
{ \
  return quantity<T, radian>(Kokkos::FUNC(q.value())); \
}

KUL_UNARY_INVERSE_TRIG_FUNCTION(asin)
KUL_UNARY_INVERSE_TRIG_FUNCTION(acos)
KUL_UNARY_INVERSE_TRIG_FUNCTION(atan)
KUL_UNARY_INVERSE_TRIG_FUNCTION(asinh)
KUL_UNARY_INVERSE_TRIG_FUNCTION(acosh)
KUL_UNARY_INVERSE_TRIG_FUNCTION(atanh)

#undef KUL_UNARY_INVERSE_TRIG_FUNCTION

template <class T>
KOKKOS_INLINE_FUNCTION constexpr auto copysign(quantity<T, unit_one> const& a, quantity<T, unit_one> const& b)
{
  return quantity<T, unit_one>(Kokkos::copysign(a.value(), b.value()));
}

template <class T>
KOKKOS_INLINE_FUNCTION constexpr auto pow(quantity<T, unit_one> const& a, quantity<T, unit_one> const& b)
{
  return quantity<T, unit_one>(Kokkos::pow(a.value(), b.value()));
}

template <class T1, class T2>
KOKKOS_INLINE_FUNCTION constexpr auto pow(quantity<T1, unit_one> const& a, T2 const& b)
{
  return kul::pow(a, quantity<T1, unit_one>(b));
}

template <class T, class Unit>
KOKKOS_INLINE_FUNCTION constexpr auto hypot(quantity<T, Unit> const& a, quantity<T, Unit> const& b)
{
  return quantity<T, Unit>(Kokkos::hypot(a.value(), b.value()));
}

template <class T, class Unit>
KOKKOS_INLINE_FUNCTION constexpr auto hypot(
    quantity<T, Unit> const& a,
    quantity<T, Unit> const& b,
    quantity<T, Unit> const& c)
{
  return quantity<T, Unit>(Kokkos::hypot(a.value(), b.value(), c.value()));
}

template <class T>
KOKKOS_INLINE_FUNCTION constexpr auto atan2(quantity<T, unit_one> const& a, quantity<T, unit_one> const& b)
{
  return quantity<T, radian>(Kokkos::atan2(a.value(), b.value()));
}

template <class T1, class Unit1,
          class T2, class Unit2,
          class T3, class Unit3>
KOKKOS_INLINE_FUNCTION constexpr auto fma(
    quantity<T1, Unit1> const& a,
    quantity<T2, Unit2> const& b,
    quantity<T3, Unit3> const& c)
{
  using T4 = decltype(a.value() * b.value() + c.value());
  using Unit4 = multiply<Unit1, Unit2>;
  static_assert(are_equal<Unit3, Unit4>,
      "fma with compile-time units: (a*b) has different units from c");
  return quantity<T4, Unit3>(Kokkos::fma(a.value(), b.value(), c.value()));
}

template <class T>
class quantity<T, dynamic_unit> {
  T m_value;
  dynamic_unit m_unit;
 public:
  using value_type = T;
  using unit_type = dynamic_unit;
  KOKKOS_DEFAULTED_FUNCTION quantity() = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr quantity(quantity&&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr quantity(quantity const&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr quantity& operator=(quantity&&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr quantity& operator=(quantity const&) = default;
  value_type const& value() const { return m_value; }
  value_type& value() { return m_value; }
  dynamic_unit const& unit() const { return m_unit; }
  quantity(T const& value_arg, dynamic_unit const& unit_arg)
    :m_value(value_arg)
    ,m_unit(unit_arg)
  {
  }
  quantity<T, dynamic_unit> in(dynamic_unit const& unit_arg) const
  {
    auto const c = conversion<T>(unit(), unit_arg);
    auto const new_value = c(value());
    return quantity<T, dynamic_unit>(new_value, unit_arg);
  }
};

// Section [named quantity]: convenience typedefs for quantities of named units

template <class T>
using seconds = quantity<T, second>;
template <class T>
using reciprocal_seconds = quantity<T, reciprocal<second>>;
template <class T>
using meters = quantity<T, meter>;
template <class T>
using square_meters = quantity<T, square_meter>;
template <class T>
using cubic_meters = quantity<T, cubic_meter>;
template <class T>
using kilograms = quantity<T, kilogram>;
template <class T>
using kelvins = quantity<T, kelvin>;
template <class T>
using temperature_electronvolts = quantity<T, temperature_electronvolt>;
template <class T>
using amperes = quantity<T, ampere>;
template <class T>
using meters_per_second = quantity<T, meter_per_second>;
template <class T>
using pascals = quantity<T, pascal>;
template <class T>
using gigapascals = quantity<T, gigapascal>;
template <class T>
using joules = quantity<T, joule>;
template <class T>
using volts = quantity<T, volt>;
template <class T>
using ohms = quantity<T, ohm>;
template <class T>
using siemens_quantity = quantity<T, siemens>;
template <class T>
using farads = quantity<T, farad>;
template <class T>
using henries = quantity<T, henry>;
template <class T>
using kilograms_per_cubic_meter = quantity<T, kilogram_per_cubic_meter>;
template <class T>
using kilogram_meters_per_second = quantity<T, kilogram_meter_per_second>;
template <class T>
using grams_per_cubic_centimeter = quantity<T, gram_per_cubic_centimeter>;
template <class T>
using joules_per_kilogram = quantity<T, joule_per_kilogram>;
template <class T>
using joules_per_kilogram_per_kelvin = quantity<T, joule_per_kilogram_per_kelvin>;
template <class T>
using megajoules_per_kilogram = quantity<T, megajoule_per_kilogram>;
template <class T>
using siemens_per_meter_quantity = quantity<T, siemens_per_meter>;

// Section [literals]: C++ user-defined literals for floating-point quantities

namespace literals {

KOKKOS_INLINE_FUNCTION constexpr
auto operator""_s(long double v)
{
  return seconds<double>(v);
}

KOKKOS_INLINE_FUNCTION constexpr
auto operator""_m(long double v)
{
  return meters<double>(v);
}

KOKKOS_INLINE_FUNCTION constexpr
auto operator""_kg(long double v)
{
  return kilograms<double>(v);
}

KOKKOS_INLINE_FUNCTION constexpr
auto operator""_kg_per_m3(long double v)
{
  return kilograms_per_cubic_meter<double>(v);
}

KOKKOS_INLINE_FUNCTION constexpr
auto operator""_K(long double v)
{
  return kelvins<double>(v);
}

KOKKOS_INLINE_FUNCTION constexpr
auto operator""_Pa(long double v)
{
  return pascals<double>(v);
}

KOKKOS_INLINE_FUNCTION constexpr
auto operator""_J(long double v)
{
  return joules<double>(v);
}

KOKKOS_INLINE_FUNCTION constexpr
auto operator""_J_per_kg(long double v)
{
  return joules_per_kilogram<double>(v);
}

KOKKOS_INLINE_FUNCTION constexpr
auto operator""_V(long double v)
{
  return volts<double>(v);
}

KOKKOS_INLINE_FUNCTION constexpr
auto operator""_Ohm(long double v)
{
  return ohms<double>(v);
}

KOKKOS_INLINE_FUNCTION constexpr
auto operator""_A(long double v)
{
  return amperes<double>(v);
}

KOKKOS_INLINE_FUNCTION constexpr
auto operator""_F(long double v)
{
  return farads<double>(v);
}

KOKKOS_INLINE_FUNCTION constexpr
auto operator""_H(long double v)
{
  return henries<double>(v);
}

}

// Section [where]: where(mask, quantity) = rhs for compatibility with SIMD

template <class M, class T, class Unit>
class const_where_expression {
 protected:
  quantity<T, Unit>& m_value;
  M m_mask;

 public:
  KOKKOS_FORCEINLINE_FUNCTION
  const_where_expression(M mask_arg, quantity<T, Unit> const& value_arg)
      : m_value(const_cast<quantity<T, Unit>&>(value_arg)), m_mask(mask_arg) {}
  KOKKOS_FORCEINLINE_FUNCTION quantity<T, Unit> const& value() const { return m_value; }
};

template <class M, class T, class Unit>
class where_expression
  : public const_where_expression<M, T, Unit> {
  using base_type = const_where_expression<M, T, Unit>;
 public:
  KOKKOS_FORCEINLINE_FUNCTION
  where_expression(M mask_arg, quantity<T, Unit>& value_arg)
      : base_type(mask_arg, value_arg) {}
  KOKKOS_FORCEINLINE_FUNCTION quantity<T, Unit>& value() { return this->m_value; }
  template <class U>
  KOKKOS_FORCEINLINE_FUNCTION void operator=(U const& x) {
    Kokkos::Experimental::where(this->m_mask, this->m_value.value()) = quantity<T, Unit>(x).value();
  }
};

template <class M, class T, class Unit>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
auto where(M const& mask, quantity<T, Unit>& value) {
  return where_expression<M, T, Unit>(mask, value);
}

template <class M, class T, class Unit>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
auto where(M const& mask, quantity<T, Unit> const& value) {
  return const_where_expression<M, T, Unit>(mask, value);
}

// Section [unit system]: representation of a runtime-defined unit system

class unit_system {
 public:
  virtual ~unit_system() = default;
  virtual dynamic_unit time_unit() const
  {
    return second();
  }
  virtual dynamic_unit length_unit() const
  {
    return meter();
  }
  virtual dynamic_unit mass_unit() const
  {
    return kilogram();
  }
  virtual dynamic_unit electric_current_unit() const
  {
    return ampere();
  }
  virtual dynamic_unit temperature_unit() const
  {
    return kelvin();
  }
  virtual dynamic_unit amount_of_substance_unit() const
  {
    return mole();
  }
  virtual dynamic_unit luminous_intensity_unit() const
  {
    return candela();
  }
  dynamic_unit unit(dimension const& d) const
  {
    dynamic_unit result = unit_one();
    for (int i = 0; i < d.time_exponent(); ++i) result *= time_unit();
    for (int i = 0; i < -d.time_exponent(); ++i) result /= time_unit();
    for (int i = 0; i < d.length_exponent(); ++i) result *= length_unit();
    for (int i = 0; i < -d.length_exponent(); ++i) result /= length_unit();
    for (int i = 0; i < d.mass_exponent(); ++i) result *= mass_unit();
    for (int i = 0; i < -d.mass_exponent(); ++i) result /= mass_unit();
    for (int i = 0; i < d.electric_current_exponent(); ++i) result *= electric_current_unit();
    for (int i = 0; i < -d.electric_current_exponent(); ++i) result /= electric_current_unit();
    for (int i = 0; i < d.amount_of_substance_exponent(); ++i) result *= amount_of_substance_unit();
    for (int i = 0; i < -d.amount_of_substance_exponent(); ++i) result /= amount_of_substance_unit();
    for (int i = 0; i < d.luminous_intensity_exponent(); ++i) result *= luminous_intensity_unit();
    for (int i = 0; i < -d.luminous_intensity_exponent(); ++i) result /= luminous_intensity_unit();
    return result;
  }
};

class si : public unit_system {
};

class esu : public unit_system {
 public:
  virtual dynamic_unit time_unit() const
  {
    return second();
  }
  virtual dynamic_unit length_unit() const
  {
    return centimeter();
  }
  virtual dynamic_unit mass_unit() const
  {
    return gram();
  }
  virtual dynamic_unit electric_current_unit() const
  {
    return statampere();
  }
};

template <class StaticUnit, class T>
quantity<T, StaticUnit> to_static(quantity<T, dynamic_unit> a)
{
  a = a.in(StaticUnit());
  return quantity<T, StaticUnit>(a.value());
}

template <class StaticUnit, class T>
quantity<T, StaticUnit> to_static(T const& a, unit_system const& s)
{
  return to_static<StaticUnit>(
      quantity<T, dynamic_unit>(
        a,
        s.unit(StaticUnit::static_dimension())));
}

template <class StaticUnit, class T>
quantity<T, StaticUnit> to_static(quantity<T, dynamic_unit> a, unit_system const& s)
{
  if (!a.unit()) return to_static<StaticUnit>(a.value(), s);
  return to_static<StaticUnit>(a);
}

}
