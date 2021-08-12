#pragma once

#include <type_traits>
#include <utility>

#include "p3a_macros.hpp"

namespace p3a {

namespace details {

template <int Index, class T>
class tuple_element {
  T m_value;
 public:
  P3A_ALWAYS_INLINE
  tuple_element() = default;
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  tuple_element(T const& value_arg)
    :m_value(value_arg)
  {}
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& get() { return m_value; }
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& get() const { return m_value; }
};

template <class TupleIndices, class ... Types>
class tuple_with_indices {
};

template <int ... Indices, class ... Types>
class tuple_with_indices<std::integer_sequence<int, Indices...>, Types...> :
  public tuple_element<Indices, Types>...
{
 public:
  using indices = std::integer_sequence<int, Indices...>;
  P3A_ALWAYS_INLINE
  tuple_with_indices() = default;
};

template <class Tuple, class Indices, class ... Types> 
class tuple_lookup_helper;

template <class Tuple, int FirstIndex, int ... NextIndices, class FirstType, class ... NextTypes>
class tuple_lookup_helper<Tuple, std::integer_sequence<int, FirstIndex, NextIndices...>, FirstType, NextTypes...>
{
 public:
  template <int LookupIndex>
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  static std::enable_if_t<LookupIndex == FirstIndex, FirstType&> lookup(Tuple& t)
  {
    return static_cast<tuple_element<FirstIndex, FirstType>*>(&t)->get();
  }
  template <int LookupIndex>
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  static auto& lookup(std::enable_if_t<LookupIndex != FirstIndex, Tuple&> t)
  {
    return tuple_lookup_helper<Tuple, std::integer_sequence<int, NextIndices...>, NextTypes...>::template lookup<LookupIndex>(t);
  }
  template <int LookupIndex>
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  static std::enable_if_t<LookupIndex == FirstIndex, FirstType const&> lookup(Tuple const& t)
  {
    return static_cast<tuple_element<FirstIndex, FirstType> const*>(&t)->get();
  }
  template <int LookupIndex>
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  static auto& lookup(std::enable_if_t<LookupIndex != FirstIndex, Tuple const&> t)
  {
    return tuple_lookup_helper<Tuple, std::integer_sequence<int, NextIndices...>, NextTypes...>::template lookup<LookupIndex>(t);
  }
};

}

template <class ... Types>
class tuple : public details::tuple_with_indices<std::make_integer_sequence<int, int(sizeof...(Types))>, Types...> {
 public:
  P3A_ALWAYS_INLINE
  tuple() = default;
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE static constexpr
  int size() { return int(sizeof...(Types)); }
  using size_constant = std::integral_constant<int, size()>;
};

template <int LookupIndex, class Indices, class ... Types> 
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto& get(details::tuple_with_indices<Indices, Types...>& t)
{
  return details::tuple_lookup_helper<
    details::tuple_with_indices<Indices, Types...>,
    Indices,
    Types...>::template lookup<LookupIndex>(t);
}

template <int LookupIndex, class Indices, class ... Types> 
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto& get(details::tuple_with_indices<Indices, Types...> const& t)
{
  return details::tuple_lookup_helper<
    details::tuple_with_indices<Indices, Types...>,
    Indices,
    Types...>::template lookup<LookupIndex>(t);
}

}
