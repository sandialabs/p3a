#include "gtest/gtest.h"
#include "p3a_lie.hpp"

using namespace p3a; 

TEST(lie, expm_spd1){
  using T = double;
  constexpr T one = 1.0;
  constexpr T zero = 0.0;
  constexpr T e = 2.718281828459045;
  symmetric3x3<T> a{one, zero, zero, one, zero, one};
  auto exp_a = spd_exponential(a);
  EXPECT_FLOAT_EQ(e, exp_a.xx()) << "exp_a.xx()";
  EXPECT_FLOAT_EQ(e, exp_a.yy()) << "exp_a.yy()";
  EXPECT_FLOAT_EQ(e, exp_a.zz()) << "exp_a.zz()";
  EXPECT_FLOAT_EQ(zero, exp_a.xy()) << "exp_a.xy()";
  EXPECT_FLOAT_EQ(zero, exp_a.xz()) << "exp_a.xz()";
  EXPECT_FLOAT_EQ(zero, exp_a.yz()) << "exp_a.yz()";
}

TEST(lie, expm_spd2){
  using T = double;
  constexpr T one = 1.0;
  constexpr T zero = 0.0;
  symmetric3x3<T> a{zero, zero, zero, zero, zero, zero};
  auto exp_a = spd_exponential(a);
  EXPECT_FLOAT_EQ(one, exp_a.xx()) << "exp_a.xx()";
  EXPECT_FLOAT_EQ(one, exp_a.yy()) << "exp_a.yy()";
  EXPECT_FLOAT_EQ(one, exp_a.zz()) << "exp_a.zz()";
  EXPECT_FLOAT_EQ(zero, exp_a.xy()) << "exp_a.xy()";
  EXPECT_FLOAT_EQ(zero, exp_a.xz()) << "exp_a.xz()";
  EXPECT_FLOAT_EQ(zero, exp_a.yz()) << "exp_a.yz()";
}

TEST(lie, expm_spd3){
  using T = double;
  symmetric3x3<T> a{T(3.0), T(1.0), T(0.0), T(2.0), T(1.0), T(1.0)};
  auto exp_a = spd_exponential(a);
  EXPECT_FLOAT_EQ(28.49937896, exp_a.xx()) << "exp_a.xx()";
  EXPECT_FLOAT_EQ(16.82033617, exp_a.yy()) << "exp_a.yy()";
  EXPECT_FLOAT_EQ(5.14129339, exp_a.zz()) << "exp_a.zz()";
  EXPECT_FLOAT_EQ(16.39468282, exp_a.xy()) << "exp_a.xy()";
  EXPECT_FLOAT_EQ(4.71564004, exp_a.xz()) << "exp_a.xz()";
  EXPECT_FLOAT_EQ(6.96340275, exp_a.yz()) << "exp_a.yz()";
  auto log_exp_a = spd_logarithm(exp_a);
  EXPECT_FLOAT_EQ(log_exp_a.xx(), a.xx()) << "exp_a.xx()";
  EXPECT_FLOAT_EQ(log_exp_a.yy(), a.yy()) << "exp_a.yy()";
  EXPECT_FLOAT_EQ(log_exp_a.zz(), a.zz()) << "exp_a.zz()";
  constexpr T tol = 8.0e-15;
  EXPECT_NEAR(log_exp_a.xy(), a.xy(), tol) << "exp_a.xy()";
  EXPECT_NEAR(log_exp_a.xz(), a.xz(), tol) << "exp_a.xz()";
  EXPECT_NEAR(log_exp_a.yz(), a.yz(), tol) << "exp_a.yz()";
}

TEST(lie, logm){
  using T = double;
  constexpr T one = 1.0;
  constexpr T zero = 0.0;
  constexpr T e = 2.718281828459045;
  symmetric3x3<T> a{e, zero, zero, e, zero, e};
  auto log_a = spd_logarithm(a);
  EXPECT_FLOAT_EQ(one, log_a.xx()) << "log_a.xx()";
  EXPECT_FLOAT_EQ(one, log_a.yy()) << "log_a.yy()";
  EXPECT_FLOAT_EQ(one, log_a.zz()) << "log_a.zz()";
  EXPECT_FLOAT_EQ(zero, log_a.xy()) << "log_a.xy()";
  EXPECT_FLOAT_EQ(zero, log_a.xz()) << "log_a.xz()";
  EXPECT_FLOAT_EQ(zero, log_a.yz()) << "log_a.yz()";
}
