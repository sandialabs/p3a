#include <gtest/gtest.h>

#include "p3a_device_ptr.hpp"

TEST(device_ptr, basic)
{
  auto uptr = p3a::make_device<int>(55);
}

class base_class {
  std::int64_t give_it_size;
 public:
  virtual P3A_HOST_DEVICE void virtual_method(double& arg) const
  {
    arg *= 2;
  }
  virtual P3A_HOST_DEVICE ~base_class()
  {
  }
};

class derived_class : public base_class {
  std::int64_t give_it_more_size;
 public:
  virtual P3A_HOST_DEVICE void virtual_method(double& arg) const override
  {
    arg = arg + arg;
  }
};

void call_virtual_method(p3a::device_ptr<base_class> const& base_ptr)
{
  auto const raw_base_ptr = base_ptr.get();
  p3a::for_each(p3a::execution::par,
      p3a::counting_iterator<int>(0),
      p3a::counting_iterator<int>(1),
  [=] P3A_HOST_DEVICE (int)
  {
    double value = 2.0;
    raw_base_ptr->virtual_method(value);
  });
}

TEST(device_ptr, derived)
{
  p3a::device_ptr<derived_class> derived_ptr = p3a::make_device<derived_class>();
  p3a::device_ptr<base_class> base_ptr{std::move(derived_ptr)};
  call_virtual_method(base_ptr);
}
