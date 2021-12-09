#include "gtest/gtest.h"
#include "p3a_lie.hpp"

using namespace p3a; 

TEST(polar_decomp, bug1){
  auto const F = matrix3x3<adimensional_quantity<double>>(
    -2.04085437594717629e+07, -1.24461560189389233e+07, -1.80950325823258795e+07,
    1.97281075824120641e+07, 1.20311933035031799e+07, 1.74917295885286182e+07,
    1.01192179273702446e+07, 6.17120657124425564e+06, 8.97210491989786923e+06);
  auto const bug_left_stretch = matrix3x3<adimensional_quantity<double>>(
    2.01426148214202598e+07, -1.96009596115295291e+07, -1.00881219945551157e+07,
    -1.96009596115295291e+07, 1.90730335562081859e+07, 9.81619113787702844e+06,
    -1.00881219945551157e+07, 9.81619113787702844e+06, 5.05197932661056146e+06);
  auto const det_F = determinant(F);
  fprintf(stderr, "det(F) %.17e\n", det_F.value());
  auto const det_bug_left_stretch = determinant(bug_left_stretch);
  fprintf(stderr, "det(buggy left stretch) %.17e\n", det_bug_left_stretch.value());
  symmetric3x3<adimensional_quantity<double>> left_stretch;
  matrix3x3<adimensional_quantity<double>> rotation;
  decompose_polar_left(F, left_stretch, rotation);
  fprintf(stderr, "got left stretch\n%.17e %.17e %.17e\n%.17e %.17e %.17e\n%.17e %.17e %.17e\n",
      left_stretch.xx().value(),
      left_stretch.xy().value(),
      left_stretch.xz().value(),
      left_stretch.yx().value(),
      left_stretch.yy().value(),
      left_stretch.yz().value(),
      left_stretch.zx().value(),
      left_stretch.zy().value(),
      left_stretch.zz().value());
  auto const det_left_stretch = determinant(left_stretch);
  fprintf(stderr, "det(left stretch) %.17e\n", det_left_stretch.value());
}

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
