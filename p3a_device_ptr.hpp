#pragma once

#include <memory>

#include <p3a_allocator.hpp>
#include <p3a_for_each.hpp>

namespace p3a {

namespace details {

template <class T>
class device_delete_functor {
  T* m_pointer;
 public:
  device_delete_functor(T* pointer_arg)
    :m_pointer(pointer_arg)
  {
  }
  P3A_HOST_DEVICE void operator()(int) const
  {
    m_pointer->~T();
  }
};

template <class T>
class device_deleter {
  std::size_t m_actual_size{sizeof(T)};
 public:
  device_deleter() = default;
  device_deleter(device_deleter&&) = default;
  device_deleter(device_deleter const&) = default;
  device_deleter& operator=(device_deleter&&) = default;
  device_deleter& operator=(device_deleter const&) = default;
  template <class U>
  device_deleter(device_deleter<U>&& other)
    :m_actual_size(other.size())
  {
  }
  void operator()(T* ptr) const
  {
    for_each(execution::par,
        counting_iterator<int>(0),
        counting_iterator<int>(1),
        device_delete_functor<T>(ptr));
    p3a::execution::par.synchronize();
    device_allocator<T> allocator;
    allocator.deallocate(ptr, m_actual_size);
  }
  std::size_t size() const { return m_actual_size; }
};

// we have to create a construct_on_device overload for each of
// a few common argument counts because CUDA does not support
// capturing a member of a parameter pack in a host/device lambda,
// nor do we have the metaprogramming skills to construct
// a functor containing the members of a parameter pack ourselves

template <class T>
void construct_on_device(T* ptr)
{
  for_each(execution::par,
      counting_iterator<int>(0),
      counting_iterator<int>(1),
  [=] P3A_HOST_DEVICE (int)
  {
    ::new (static_cast<void*>(ptr)) T();
  });
}

template <class T, class Arg1>
void construct_on_device(T* ptr, Arg1 arg1)
{
  for_each(execution::par,
      counting_iterator<int>(0),
      counting_iterator<int>(1),
  [=] P3A_HOST_DEVICE (int)
  {
    ::new (static_cast<void*>(ptr)) T(arg1);
  });
}

template <class T, class Arg1, class Arg2>
void construct_on_device(T* ptr, Arg1 arg1, Arg2 arg2)
{
  for_each(execution::par,
      counting_iterator<int>(0),
      counting_iterator<int>(1),
  [=] P3A_HOST_DEVICE (int)
  {
    ::new (static_cast<void*>(ptr)) T(arg1, arg2);
  });
}

template <class T, class Arg1, class Arg2, class Arg3>
void construct_on_device(T* ptr, Arg1 arg1, Arg2 arg2, Arg3 arg3)
{
  for_each(execution::par,
      counting_iterator<int>(0),
      counting_iterator<int>(1),
  [=] P3A_HOST_DEVICE (int)
  {
    ::new (static_cast<void*>(ptr)) T(arg1, arg2, arg3);
  });
}

}

template <class T>
using device_ptr = std::unique_ptr<T, details::device_deleter<T>>;

template <class T, class... Args>
device_ptr<T> make_device(Args... args)
{
  device_allocator<T> allocator;
  T* raw_pointer = allocator.allocate(sizeof(T));
  details::construct_on_device(raw_pointer, args...);
  return device_ptr<T>(raw_pointer);
}

}

