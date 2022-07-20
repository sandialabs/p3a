#pragma once

#include <stdexcept>

#include <Kokkos_StdAlgorithms.hpp>

#include "p3a_execution.hpp"

namespace p3a {

namespace details {

template <class Iterator1, class Iterator2, class T>
class kokkos_exclusive_scan_functor {
  Iterator1 m_first;
  Iterator2 m_d_first;
 public:
  kokkos_exclusive_scan_functor(
      Iterator1 first_arg,
      Iterator2 d_first_arg)
    :m_first(first_arg)
    ,m_d_first(d_first_arg)
  {
  }
  using difference_type = typename std::iterator_traits<Iterator1>::difference_type;
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
  void operator()(difference_type const i, T& update, bool const is_final_pass) const
  {
    if (is_final_pass) {
      m_d_first[i] = update;
    }
    update += m_first[i];
  }
};

template <
  class ExecutionSpace,
  class Iterator1,
  class Iterator2,
  class T>
void kokkos_exclusive_scan(
    Iterator1 first,
    Iterator1 last,
    Iterator2 d_first,
    T init)
{
  if (init != T(0)) {
    throw std::runtime_error("p3a::details::kokkos_exclusive_scan only supports zero init");
  }
  using difference_type = typename std::iterator_traits<Iterator1>::difference_type;
  using kokkos_policy =
    Kokkos::RangePolicy<
      ExecutionSpace,
      Kokkos::IndexType<difference_type>>;
  using functor = kokkos_exclusive_scan_functor<Iterator1, Iterator2, T>;
  Kokkos::parallel_scan("p3a::details::kokkos_exclusive_scan",
      kokkos_policy(0, (last - first)),
      functor(first, d_first));
}

template <class InputIt, class OutputIt, class UnaryPredicate>
class kokkos_copy_if_functor {
  InputIt m_first;
  OutputIt m_d_first;
  UnaryPredicate m_pred;
 public:
  kokkos_copy_if_functor(
      InputIt first_arg,
      OutputIt d_first_arg,
      UnaryPredicate pred_arg)
    :m_first(first_arg)
    ,m_d_first(d_first_arg)
    ,m_pred(pred_arg)
  {
  }
  using input_difference_type = typename std::iterator_traits<InputIt>::difference_type;
  using output_difference_type = typename std::iterator_traits<OutputIt>::difference_type;
  P3A_ALWAYS_INLINE P3A_HOST_DEVICE inline
  void operator()(
      input_difference_type const i,
      output_difference_type& update,
      bool const is_final_pass) const
  {
    if (is_final_pass) {
      if (m_pred(m_first[i])) {
        m_d_first[update] = m_first[i];
      }
    }
    if (m_pred(m_first[i])) {
      update += 1;
    }
  }
};

template <
  class ExecutionSpace,
  class InputIt,
  class OutputIt,
  class UnaryPredicate>
OutputIt kokkos_copy_if(
    InputIt first,
    InputIt last,
    OutputIt d_first,
    UnaryPredicate pred)
{
  using input_difference_type = typename std::iterator_traits<InputIt>::difference_type;
  using kokkos_policy =
    Kokkos::RangePolicy<
      ExecutionSpace,
      Kokkos::IndexType<input_difference_type>>;
  using functor = kokkos_copy_if_functor<InputIt, OutputIt, UnaryPredicate>;
  input_difference_type copied_count = 0;
  Kokkos::parallel_scan("p3a::details::kokkos_copy_if",
      kokkos_policy(0, (last - first)),
      functor(first, d_first, pred),
      copied_count);
  return d_first + copied_count;
}

}

template <
  class ExecutionPolicy,
  class Iterator1,
  class Iterator2,
  class T>
void exclusive_scan(
    ExecutionPolicy policy,
    Iterator1 first,
    Iterator1 last,
    Iterator2 d_first,
    T init)
{
  details::kokkos_exclusive_scan<typename ExecutionPolicy::kokkos_execution_space>(
      first, last, d_first, init);
}

template <
  class ExecutionPolicy,
  class InputIt,
  class OutputIt,
  class UnaryPredicate>
OutputIt copy_if(
    ExecutionPolicy policy,
    InputIt first,
    InputIt last,
    OutputIt d_first,
    UnaryPredicate pred)
{
  return details::kokkos_copy_if<typename ExecutionPolicy::kokkos_execution_space>(
      first, last, d_first, pred);
}

}
