#include "gtest/gtest.h"
#include "p3a_exp.hpp"
#include "p3a_lie.hpp"
#include "p3a_log.hpp"

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

TEST(tensor, exp)
{
  using Real   = double;
  using Tensor = matrix3x3<Real>;
  // Identity
  auto const eps = epsilon_value<Real>();
  auto const I   = Tensor::identity();
  auto const z   = Tensor(0, 0, 0, 0, 0, 0, 0, 0, 0);
  auto const Z   = exp(z);
  auto const error_Z = norm(Z - I) / norm(I);
  ASSERT_LE(error_Z, eps);
  auto const A          = Tensor(2.5, 0.5, 1, 0.5, 2.5, 1, 1, 1, 2);
  auto const a          = sqrt(2.0) / 2.0;
  auto const b          = sqrt(3.0) / 3.0;
  auto const c          = sqrt(6.0) / 6.0;
  auto const V          = Tensor(c, a, b, c, -a, b, -2 * c, 0, b);
  auto const p          = std::exp(1.0);
  auto const q          = std::exp(2.0);
  auto const r          = std::exp(4.0);
  auto const D          = Tensor(p, 0, 0, 0, q, 0, 0, 0, r);
  auto const B          = exp(A);
  auto const C          = V * D * transpose(V);
  auto const F          = exp_taylor(A);
  auto const error_pade = norm(B - C) / norm(C);
  auto const tol        = 3 * eps;
  ASSERT_LE(error_pade, tol);
  auto const error_taylor = norm(F - C) / norm(C);
  ASSERT_LE(error_taylor, tol);
}

TEST(tensor, log)
{
  using Real   = double;
  using Tensor = matrix3x3<Real>;
  // Identity
  auto const eps     = epsilon_value<Real>();
  auto const I       = Tensor::identity();
  auto const i       = log(I);
  auto const error_I = norm(i) / norm(I);
  ASSERT_LE(error_I, eps);
  // 1/8 of a rotation
  auto const tau     = 2.0 * std::acos(-1.0);
  auto const c       = std::sqrt(2.0) / 2.0;
  auto const R       = Tensor(c, -c, 0.0, c, c, 0.0, 0.0, 0.0, 1.0);
  auto const r       = log(R);
  auto const error_R = std::abs(r(0, 1) + tau / 8.0);
  ASSERT_LE(error_R, eps);
  auto const error_r = std::abs(r(0, 1) + r(1, 0));
  ASSERT_LE(error_r, eps);
  auto const A       = Tensor(7, 1, 2, 3, 8, 4, 5, 6, 9);
  auto const a       = log(A);
  auto const b       = log_gregory(A);
  auto const error_a = norm(b - a);
  auto const tol     = 20 * eps;
  ASSERT_LE(error_a, tol);
}
