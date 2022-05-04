#pragma once

#include "Kokkos_Core.hpp"

#include "p3a_for_each.hpp"

namespace p3a {

template <class T>
class simd_view {
 private:
  using value_t = typename Kokkos::View<T>::value_type;
  using traits_t = typename Kokkos::View<T>::traits;
  using specialize_t = typename Kokkos::View<T>::specialize;
  using map_t = Kokkos::Impl::ViewMapping<traits_t, specialize_t>;
  template <class Abi> using simd_t = simd<value_t, Abi>;
  template <class Abi> using mask_t = simd_mask<value_t, Abi>;
 private:
  Kokkos::View<T> m_view;
  map_t m_map;
  value_t* m_data = nullptr;
 public:
  simd_view() = default;
  simd_view(Kokkos::View<T> view)
    : m_view(view), m_map(view.impl_map()), m_data(view.data())
  {}
  template <class Abi>
  [[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  typename std::enable_if<1 == Kokkos::View<T>::Rank, simd_t<Abi>>::type
  load(int i, mask_t<Abi> const& mask) const {
    return p3a::load(m_data, i, mask);
  }
  template <class Abi>
  [[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  typename std::enable_if<2 == Kokkos::View<T>::Rank, simd_t<Abi>>::type
  load(int i, int j, mask_t<Abi> const& mask) const {
    int const idx = m_map.m_impl_offset(i,j);
    return p3a::load(m_data, idx, mask);
  }
  template <class Abi>
  [[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  typename std::enable_if<3 == Kokkos::View<T>::Rank, simd_t<Abi>>::type
  load(int i, int j, int k, mask_t<Abi> const& mask) const {
    int const idx = m_map.m_impl_offset(i,j,k);
    return p3a::load(m_data, idx, mask);
  }
  template <class Abi>
  [[nodiscard]] P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  typename std::enable_if<4 == Kokkos::View<T>::Rank, simd_t<Abi>>::type
  load(int i, int j, int k, int l, mask_t<Abi> const& mask) const {
    int const idx = m_map.m_impl_offset(i,j,k,l);
    return p3a::load(m_data, idx, mask);
  }
  template <class Abi, class U = T>
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  typename std::enable_if<1 == Kokkos::View<U>::Rank>::type
  store(simd_t<Abi> const& val, int i, mask_t<Abi> const& mask) const {
    p3a::store(val, m_data, i, mask);
  }
  template <class Abi, class U = T>
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  typename std::enable_if<2 == Kokkos::View<U>::Rank>::type
  store(simd_t<Abi> const& val, int i, int j, mask_t<Abi> const& mask) const {
    int const idx = m_map.m_impl_offset(i,j);
    p3a::store(val, m_data, idx, mask);
  }
  template <class Abi, class U = T>
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  typename std::enable_if<3 == Kokkos::View<U>::Rank>::type
  store(simd_t<Abi> const& val, int i, int j, int k, mask_t<Abi> const& mask) const {
    int const idx = m_map.m_impl_offset(i,j,k);
    p3a::store(val, m_data, idx, mask);
  }
  template <class Abi, class U = T>
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  typename std::enable_if<4 == Kokkos::View<U>::Rank>::type
  store(simd_t<Abi> const& val, int i, int j, int k, int l, mask_t<Abi> const& mask) const {
    int const idx = m_map.m_impl_offset(i,j,k,l);
    p3a::store(val, m_data, idx, mask);
  }
  template <class Abi, class U = T>
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  typename std::enable_if<1 == Kokkos::View<U>::Rank>::type
  sum_store(simd_t<Abi> const& val, int i, mask_t<Abi> const& mask) const {
    simd_t<Abi> sum = load(i, mask);
    sum += val;
    store(sum, i, mask);
  }
  template <class Abi, class U = T>
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  typename std::enable_if<2 == Kokkos::View<U>::Rank>::type
  sum_store(simd_t<Abi> const& val, int i, int j, mask_t<Abi> const& mask) const {
    simd_t<Abi> sum = load(i, j, mask);
    sum += val;
    store(sum, i, j, mask);
  }
  template <class Abi, class U = T>
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  typename std::enable_if<3 == Kokkos::View<U>::Rank>::type
  sum_store(simd_t<Abi> const& val, int i, int j, int k, mask_t<Abi> const& mask) const {
    simd_t<Abi> sum = load(i, j, k, mask);
    sum += val;
    store(sum, i, j, k, mask);
  }
  template <class Abi, class U = T>
  P3A_ALWAYS_INLINE P3A_HOST P3A_DEVICE inline
  typename std::enable_if<4 == Kokkos::View<U>::Rank>::type
  sum_store(simd_t<Abi> const& val, int i, int j, int k, int l, mask_t<Abi> const& mask) const {
    simd_t<Abi> sum = load(i, j, k, l, mask);
    sum += val;
    store(sum, i, j, k, l, mask);
  }
};

}
