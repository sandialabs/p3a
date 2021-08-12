#include "p3a_tuple.hpp"
#include "p3a_for_each.hpp"

#include <type_traits>
#include <iostream>

class functor {
 public:
  using tuple_type = p3a::tuple<int, double>;
  tuple_type the_tuple;
  functor(tuple_type const& tuple_arg)
    :the_tuple(tuple_arg)
  {}
  template <int i>
  void operator()(std::integral_constant<int, i>) const
  {
    std::cout << p3a::get<i>(the_tuple) << '\n';
  }
};

int main() {
  p3a::tuple<int, double> const t{22, 3.14};
  p3a::for_each(p3a::serial, decltype(t)::size_constant(), functor(t));
}
