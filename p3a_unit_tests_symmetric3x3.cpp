#include "gtest/gtest.h"
#include "p3a_symmetric3x3.hpp"

using namespace p3a; 

TEST(symmetric3x3, isotropic_part){
  using T = double;
  T const zero = T(0.);
  T const one = T(1.);
  T const two = T(2.);
  T const three = T(3.);
  T const six = T(6.);
  symmetric3x3<T> a{one, zero, zero, one, zero, one};
  symmetric3x3<T> b;
  b = isotropic_part(a);
  EXPECT_FLOAT_EQ(trace(b), trace(a));
  EXPECT_FLOAT_EQ(b.xx(), one);
  EXPECT_FLOAT_EQ(b.yy(), one);
  EXPECT_FLOAT_EQ(b.zz(), one);
  a.yy() = two;
  a.zz() = three;
  b = isotropic_part(a);
  EXPECT_FLOAT_EQ(trace(b), trace(a));
  EXPECT_FLOAT_EQ(b.xx(), six / three);
  EXPECT_FLOAT_EQ(b.yy(), b.xx());
  EXPECT_FLOAT_EQ(b.zz(), b.xx());
}

TEST(symmetric3x3, deviatoric_part){
  using T = double;
  T const zero = T(0.);
  T const one = T(1.);
  T const two = T(2.);
  T const three = T(3.);
  T const six = T(6.);
  symmetric3x3<T> a{one, one, one, one, one, one};
  symmetric3x3<T> b;
  b = deviatoric_part(a);
  EXPECT_FLOAT_EQ(trace(b), zero);
  EXPECT_FLOAT_EQ(b.xx(), zero);
  EXPECT_FLOAT_EQ(b.yy(), zero);
  EXPECT_FLOAT_EQ(b.zz(), zero);
  EXPECT_FLOAT_EQ(b.xy(), one);
  EXPECT_FLOAT_EQ(b.xz(), one);
  EXPECT_FLOAT_EQ(b.yz(), one);
  a.yy() = two;
  a.zz() = three;
  b = deviatoric_part(a);
  EXPECT_FLOAT_EQ(trace(b), zero);
  EXPECT_FLOAT_EQ(b.xx(), one - six / three);
  EXPECT_FLOAT_EQ(b.yy(), two - six / three);
  EXPECT_FLOAT_EQ(b.zz(), three - six / three);
}

TEST(symmetric3x3, inverse_diag){
  using T = double;
  T const zero = T(0.);
  T const one = T(1.);
  T const two = T(2.);
  symmetric3x3<T> a{two, zero, zero, two, zero, two};
  auto ai = inverse(a);
  EXPECT_FLOAT_EQ(ai.xx(), one / two);
  EXPECT_FLOAT_EQ(ai.yy(), one / two);
  EXPECT_FLOAT_EQ(ai.zz(), one / two);
  EXPECT_FLOAT_EQ(ai.xy(), zero);
  EXPECT_FLOAT_EQ(ai.xz(), zero);
  EXPECT_FLOAT_EQ(ai.yz(), zero);
}

TEST(symmetric3x3, inverse){
  using T = double;
  symmetric3x3<T> a{1., .5, .1, 2., .2, 3.};
  auto ai = inverse(a);
  EXPECT_FLOAT_EQ(ai.xx(), 1.143953934740883);
  EXPECT_FLOAT_EQ(ai.yy(), 0.5738963531669865);
  EXPECT_FLOAT_EQ(ai.zz(), 0.33589251439539347);
  EXPECT_FLOAT_EQ(ai.xy(), -0.2840690978886756);
  EXPECT_FLOAT_EQ(ai.xz(), -0.019193857965451054);
  EXPECT_FLOAT_EQ(ai.yz(), -0.028790786948176588);
}
