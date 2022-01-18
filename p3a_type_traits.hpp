#pragma once

namespace p3a {

template <class T>
class no_deduce {
 public:
  using type = T;
};

template <class T>
using no_deduce_t = typename no_deduce<T>::type;

}
