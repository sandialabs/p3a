#include "gtest/gtest.h"
#include "p3a_lie.hpp"

#include "p3a_counting_iterator.hpp"
#include "p3a_for_each.hpp"

using namespace p3a;

TEST(polar_decomp, stretch){
  using T = double;
  T const l = std::sqrt(T(2.));
  matrix3x3<T> F{l, 0.0, 0.0, 0.0, 2 * l, 0.0, 0.0, 0.0, 1.0};
  matrix3x3<T> R;
  symmetric3x3<T> U;

  auto const e = decompose_polar_right(F, R, U);
  EXPECT_EQ(e, polar_errc::success);

  EXPECT_FLOAT_EQ(l, U.xx()) << "U.xx()";
  EXPECT_FLOAT_EQ(2 * l, U.yy()) << "U.yy()";
  EXPECT_FLOAT_EQ(1.0, U.zz()) << "U.zz()";
  EXPECT_FLOAT_EQ(0.0, U.xy()) << "U.xy()";
  EXPECT_FLOAT_EQ(0.0, U.xz()) << "U.xz()";
  EXPECT_FLOAT_EQ(0.0, U.yz()) << "U.yz()";

  EXPECT_FLOAT_EQ(1.0, R.xx()) << "R.xx()";
  EXPECT_FLOAT_EQ(1.0, R.yy()) << "R.yy()";
  EXPECT_FLOAT_EQ(1.0, R.zz()) << "R.zz()";
  EXPECT_FLOAT_EQ(0.0, R.xy()) << "R.xy()";
  EXPECT_FLOAT_EQ(0.0, R.xz()) << "R.xz()";
  EXPECT_FLOAT_EQ(0.0, R.yx()) << "R.yx()";
  EXPECT_FLOAT_EQ(0.0, R.yz()) << "R.yz()";
  EXPECT_FLOAT_EQ(0.0, R.zx()) << "R.zx()";
  EXPECT_FLOAT_EQ(0.0, R.zy()) << "R.zy()";

}

TEST(polar_decomp, pure_shear){
  using T = double;
  T const l = std::sqrt(T(2.));
  matrix3x3<T> F{l, 0.0, 0.0, 0.0, 1. / l, 0.0, 0.0, 0.0, T(1.0)};
  matrix3x3<T> R;
  symmetric3x3<T> U;

  auto const e = decompose_polar_right(F, R, U);
  EXPECT_EQ(e, polar_errc::success);

  EXPECT_FLOAT_EQ(l, U.xx()) << "U.xx()";
  EXPECT_FLOAT_EQ(1./l, U.yy()) << "U.yy()";
  EXPECT_FLOAT_EQ(1., U.zz()) << "U.zz()";
  EXPECT_FLOAT_EQ(0.0, U.xy()) << "U.xy()";
  EXPECT_FLOAT_EQ(0.0, U.xz()) << "U.xz()";
  EXPECT_FLOAT_EQ(0.0, U.yz()) << "U.yz()";

  EXPECT_FLOAT_EQ(1.0, R.xx()) << "R.xx()";
  EXPECT_FLOAT_EQ(0.0, R.xy()) << "R.xy()";
  EXPECT_FLOAT_EQ(0.0, R.xz()) << "R.xz()";

  EXPECT_FLOAT_EQ(0.0, R.yx()) << "R.yx()";
  EXPECT_FLOAT_EQ(1.0, R.yy()) << "R.yy()";
  EXPECT_FLOAT_EQ(0.0, R.yz()) << "R.yz()";

  EXPECT_FLOAT_EQ(0.0, R.zx()) << "R.zx()";
  EXPECT_FLOAT_EQ(0.0, R.zy()) << "R.zy()";
  EXPECT_FLOAT_EQ(1.0, R.zz()) << "R.zz()";
}


TEST(polar_decomp, simple_shear){
  using T = double;
  T const zero = T(0.);
  T const one = T(1.);
  T const two = T(2.);
  T const root2 = std::sqrt(T(2.));
  T const root3 = std::sqrt(T(3.));
  T const toor3 = one / root3;
  T const root23 = root2 / root3;

  matrix3x3<T> F{one, root2, zero, zero, one, zero, zero, zero, one};
  matrix3x3<T> R;
  symmetric3x3<T> U;

  auto const e = decompose_polar_right(F, R, U);
  EXPECT_EQ(e, polar_errc::success);

  EXPECT_FLOAT_EQ(root23, U.xx()) << "U.xx()";
  EXPECT_FLOAT_EQ(two * root23, U.yy()) << "U.yy()";
  EXPECT_FLOAT_EQ(one, U.zz()) << "U.zz()";
  EXPECT_FLOAT_EQ(toor3, U.xy()) << "U.xy()";
  EXPECT_FLOAT_EQ(zero, U.xz()) << "U.xz()";
  EXPECT_FLOAT_EQ(zero, U.yz()) << "U.yz()";

  EXPECT_FLOAT_EQ(root23, R.xx()) << "R.xx()";
  EXPECT_FLOAT_EQ(toor3, U.xy()) << "R.xy()";
  EXPECT_FLOAT_EQ(zero, R.xz()) << "R.xz()";

  EXPECT_FLOAT_EQ(-toor3, R.yx()) << "R.yx()";
  EXPECT_FLOAT_EQ(root23, R.yy()) << "R.yy()";
  EXPECT_FLOAT_EQ(zero, R.yz()) << "R.yz()";

  EXPECT_FLOAT_EQ(zero, R.zx()) << "R.zx()";
  EXPECT_FLOAT_EQ(zero, R.zy()) << "R.zy()";
  EXPECT_FLOAT_EQ(one, R.zz()) << "R.zz()";
}
