#pragma once

#include "p3a_scalar.hpp"
#include "p3a_macros.hpp"
#include "p3a_functions.hpp"
#include "p3a_diagonal3x3.hpp"
#include "p3a_constants.hpp"
#include "p3a_vector3.hpp"
#include "p3a_symmetric3x3.hpp"
#include "p3a_static_matrix.hpp"
#include "p3a_matrix3x3.hpp"
/********************************* NOTES ************************************
 *
 * Symmetric Tensor (Voigt-Mandel) Math Library
 * ============================================
 *
 * Scope of the Library
 * ----------------------------------
 * This set of classes and functions allows for linear algebra to be performed
 * in the context of material models. This package enables the usual linear
 * algebra operations with symmetric 1x6, 6x1, 6x6, 6x3, and 3x6 tensors and 
 * is able to perform mixed operations with the `diagonal3x3`, `symmetric3x3` 
 * and `matrix3x3` classes also avaliable in P3A.  operations are carried out 
 * component by component, elimininating the need for arrays and permitting 
 * functions to be _inline_ consistent with the performance requirements of
 * P3A. 
 *
 * This header provides the class: 
 *
 * - `mandel6x1` (6x1) 2nd order Tensor
 *   
 *   Constructors:
 *   
 *   - `mandel6x1(mandel6x1)`
 *   - `mandel6x1(<list of values>)`
 *   - `mandel6x1(symmetric3x3)`
 *   - `mandel6x1(matrix3x3)` -- includes testing for symmetry
 *   - `mandel6x1(static_matrix3x3)` -- includes testing for symmetry
 *
 * Other `mandelNxN` headers provide the classes:
 *
 * - `mandel6x6` (6x6) 4th order Tensor
 * - `mandel3x6` (3x6) 3rd order Tensor
 * - `mandel6x3` (6x3) 3rd order Tensor
 *
 * Mandel Notation and Voigt Notation
 * ----------------------------------
 * The primary addition of this library is the ability to perform the usual
 * linear algebra relations with normal Tensors and Vectors along with symmetric 
 * ones often found in mechanics. This package uses *Mandel* notation, not 
 * _Voigt_ notation. Mandel notation is simmilar to Voigt's notation in that
 * symmetric 2<sup>nd</sup>-order tensors are represented by 6x1 vectors, and 
 * 4<sup>th</sup>-order tensors are represented by 6x6 tensors. Mandel notation 
 * is applied to all tensors/vectors unlike Voigt notation. 
 *
 * Mandel notation allows for all standard linear algebra operations to be
 * correctly normalized and eliminates the need for strain, or stress-like Voigt
 * tensors (symmetric strain-like tensors will have of 2 in front of off-diagonal 
 * components, while stress like tensors will have a factor of 1). Converting 
 * to Mandel notation (which properly normalizes the tensor) will ease computations
 * with operations produce vectors (3x1) or full (3x3) 2<sup>nd</sup>-order tensors.
 *
 * Mandel Transformation is applied internally upon construction of a Mandel-type 
 * object which includes all of the symmetric Mandel Tensor types (6x1, 6x6, 3x6, 
 * or 6x3). Avaliable constructors vary by mandel tensor type, so see notes in each
 * specific header. By default, all constructors apply the Mandel Transform, but it can be 
 * overridden by specifying a Boolean `false` value to the end of the constructor 
 * argument list. Note that, if you use the `zero()` or `identity()` constructor, 
 * you will have a tensor that won't carry the transform to further operations, the 
 * transformation must be applied manually by applying the method `MandelXform()` 
 * method to the Mandel-type object. The transform will be contained in any result 
 * returning a Mandel-type object. This means that the Mandel transformation must be 
 * inverted when converting mandel6x1 (6-element symmetric tensors) to full 
 * 9-element (3x3) 2<sup>nd</sup>-order tensors. 
 *
 * This library automatically inverts the Mandel transformation when returning a
 * member of the `matrix3x3` or `symmetric3x3` classes. Operations that 
 * return Vectors will not need to be modified (this is why we are using Mandel 
 * transformations)!  
 *
 */

namespace p3a {


/******************************************************************/
/******************************************************************/
template <class T>
class mandel6x1
/** 
 * Represents a 2nd order tensor as a 6x1 Mandel array
 * 
 ******************************************************************/
{
 T m_x1,m_x2,m_x3,m_x4,m_x5,m_x6;
 bool applyTransform;

 public:
  static constexpr T r2 = square_root_of_two_value<T>();

  static constexpr T r2i = T(1.0)/square_root_of_two_value<T>();

  static constexpr T two= T(2.0);

  /**** constructors, destructors, and assigns ****/
  P3A_ALWAYS_INLINE constexpr
  mandel6x1() = default;

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  mandel6x1(
      T const& X1, T const& X2, T const& X3,
      T const& X4, T const& X5, T const& X6):
       m_x1(X1),
       m_x2(X2),
       m_x3(X3),
       m_x4(X4),
       m_x5(X5),
       m_x6(X6),
       applyTransform(true)
  {
    this->MandelXform();
  }

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  mandel6x1(
      T const& X1, T const& X2, T const& X3,
      T const& X4, T const& X5, T const& X6, bool const& Xform)
    :m_x1(X1)
    ,m_x2(X2)
    ,m_x3(X3)
    ,m_x4(X4)
    ,m_x5(X5)
    ,m_x6(X6)
    ,applyTransform(Xform)
  {
    if (applyTransform)
        this->MandelXform();
  }

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  mandel6x1(symmetric3x3<T> const& a):
    m_x1(a.xx()),
    m_x2(a.yy()),
    m_x3(a.zz()),
    m_x4(a.yz()),
    m_x5(a.xz()),
    m_x6(a.xy()),
    applyTransform(true)
  {
    this->MandelXform();
  }

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  mandel6x1(symmetric3x3<T> const& a, bool const& Xform):
    m_x1(a.xx()),
    m_x2(a.yy()),
    m_x3(a.zz()),
    m_x4(a.yz()),
    m_x5(a.xz()),
    m_x6(a.xy()),
    applyTransform(Xform)
  {
    if (applyTransform)
        this->MandelXform();
  }

  P3A_HOST P3A_DEVICE P3A_NEVER_INLINE
  mandel6x1(matrix3x3<T> const& a):
    m_x1(a.xx()),
    m_x2(a.yy()),
    m_x3(a.zz()),
    m_x4(a.yz()),
    m_x5(a.xz()),
    m_x6(a.xy()),
    applyTransform(true)
  {
    this->MandelXform();
  }

  P3A_NEVER_INLINE
  mandel6x1(matrix3x3<T> const& a, bool const& Xform):
    m_x1(a.xx()),
    m_x2(a.yy()),
    m_x3(a.zz()),
    m_x4(a.yz()),
    m_x5(a.xz()),
    m_x6(a.xy()),
    applyTransform(Xform)
  {
    if(!compare(a.yz(),a.zy()) && compare(a.zx(),a.xz()) && compare(a.xy(),a.yx()))
        throw std::invalid_argument(
                "Initialization ERROR of p3a::mandel6x6 from p3a::matrix3x3, matrix3x3 not symmetric!");
    if (applyTransform)
        this->MandelXform();
  }

  P3A_NEVER_INLINE
  mandel6x1(static_matrix<T,3,3> const& a):
    m_x1(a(0,0)),
    m_x2(a(1,1)),
    m_x3(a(2,2)),
    m_x4(a(1,2)),
    m_x5(a(0,2)),
    m_x6(a(0,1)),
    applyTransform(true)
  {
    if(!compare(a(1,2),a(2,1)) && compare(a(0,2),a(2,0)) && compare(a(0,1),a(1,0)))
        throw std::invalid_argument(
                "Initialization ERROR of p3a::mandel6x1 from p3a::static_matrix<3,3>, static_matrix<3,3> not symmetric!");
    this->MandelXform();
  }

  P3A_NEVER_INLINE
  mandel6x1(static_matrix<T,3,3> const& a, bool const& Xform):
    m_x1(a(0,0)),
    m_x2(a(1,1)),
    m_x3(a(2,2)),
    m_x4(a(1,2)),
    m_x5(a(0,2)),
    m_x6(a(0,1)),
    applyTransform(Xform)
  {
    if(!compare(a(1,2),a(2,1)) && compare(a(0,2),a(2,0)) && compare(a(0,1),a(1,0)))
        throw std::invalid_argument(
                "Initialization ERROR of p3a::mandel6x1 from p3a::static_matrix<3,3>, static_matrix<3,3> not symmetric!");
    if (applyTransform)
        this->MandelXform();
  }


  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  mandel6x1(diagonal3x3<T> const& a):
    m_x1(a.xx()),
    m_x2(a.yy()),
    m_x3(a.zz()),
    m_x4(T(0.0)),
    m_x5(T(0.0)),
    m_x6(T(0.0)),
    applyTransform(true)
  {
    this->MandelXform();
  }

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  mandel6x1(diagonal3x3<T> const& a, bool const& Xform):
    m_x1(a.xx()),
    m_x2(a.yy()),
    m_x3(a.zz()),
    m_x4(T(0.0)),
    m_x5(T(0.0)),
    m_x6(T(0.0)),
    applyTransform(Xform)
  {
      if(applyTransform)
          this->MandelXform();
  }

  //Return components by ij descriptor
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& xx() const { return m_x1; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& yy() const { return m_x2; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& zz() const { return m_x3; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& yz() const { return m_x4; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& xz() const { return m_x5; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& xy() const { return m_x6; }

  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& xx() { return m_x1; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& yy() { return m_x2; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& zz() { return m_x3; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& yz() { return m_x4; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& xz() { return m_x5; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& xy() { return m_x6; }

  //return by mandel index 1-6
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x1() const { return m_x1; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x2() const { return m_x2; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x3() const { return m_x3; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x4() const { return m_x4; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x5() const { return m_x5; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x6() const { return m_x6; }

  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x1() { return m_x1; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x2() { return m_x2; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x3() { return m_x3; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x4() { return m_x4; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x5() { return m_x5; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x6() { return m_x6; }

  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE static constexpr
  mandel6x1<T> zero()
  {
    return mandel6x1<T>(
        T(0), T(0), T(0),
        T(0), T(0), T(0),false);
  }

  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE static constexpr
  mandel6x1<T> identity()
  {
    return mandel6x1<T>(
        T(1), T(1), T(1),
        T(0), T(0), T(0));
  }

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  void MandelXform()
  {
      m_x4 *= r2;
      m_x5 *= r2;
      m_x6 *= r2;
  }

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  void invMandelXform()
  {
      m_x4 /= r2;
      m_x5 /= r2;
      m_x6 /= r2;
  }

  //conversion of symmetric3x3 to mandel6x1 via assignment
  template <class U>
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  mandel6x1<U> operator=(
      symmetric3x3<U> const& t)
  {
      return mandel6x1<U>(
              t.xx(),
              t.yy(),
              t.zz(),
              t.yz(),
              t.xz(),
              t.xy(),
              true);
  }

  //conversion of static_matrix<3,3> to mandel6x1 via assignment
  template <class U>
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  mandel6x1<U> operator=(
      static_matrix<U,3,3> const& t)
  {
        /*mandel6x1<U>(t(0,0), t(1,1), t(2,2), t(1,2), t(0,2), t(0,1), true);
    if(compare(a(1,2),a(2,1)) && compare(a(0,2),a(2,0)) && compare(a(0,1),a(1,0)))
        throw std::invalid_argument(
                "Initialization ERROR of p3a::mandel6x1 from p3a::static_matrix<3,3>, static_matrix<3,3> not symmetric!");
        this->MandelXform();
  }*/
      return mandel6x1<U>(t , true);
  }

};

/***************************************************************************** 
 * Operators overloads for mandel6x1 tensors (2nd order tensor)
 *****************************************************************************/

//mandel6x1 binary operators with scalars
//multiplication by constant
template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<B>, mandel6x1<decltype(A() * B())>>::type
operator*(
    mandel6x1<A> const& t, 
    B const& c)
{
    return mandel6x1<decltype(t.x1()*c)>(
            t.x1()*c,
            t.x2()*c,
            t.x3()*c,
            t.x4()*c,
            t.x5()*c,
            t.x6()*c,
            false);
}

//multiplication by constant
template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<A>, mandel6x1<decltype(A() * B())>>::type 
operator*(
    A const& c, 
    mandel6x1<B> const& t)
{
    return t * c;
}

//division by constant
template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<B>, mandel6x1<decltype(A() * B())>>::type 
operator/(
    mandel6x1<A> const& t, 
    B const& c)
{
    return mandel6x1<decltype(A() * B())>(
            t.x1() / c,
            t.x2() / c,
            t.x3() / c,
            t.x4() / c,
            t.x5() / c,
            t.x6() / c,
            false);
}

//multiplication *= by constant
template <class A>
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
void operator*=(
    mandel6x1<A>& t, 
    A const& c)
{
    t.x1() *= c;
    t.x2() *= c;
    t.x3() *= c;
    t.x4() *= c;
    t.x5() *= c;
    t.x6() *= c;
}

//division /= by constant
template <class A>
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
void operator/=(
    mandel6x1<A>& t, 
    A const& c)
{
    t.x1() /= c;
    t.x2() /= c;
    t.x3() /= c;
    t.x4() /= c;
    t.x5() /= c;
    t.x6() /= c;
}

//mandel6x1 -= subtraction
template <class T>
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
void operator-=(
    mandel6x1<T>& a, 
    mandel6x1<T> const& b)
{
    a.x1() -= b.x1();
    a.x2() -= b.x2();
    a.x3() -= b.x3();
    a.x4() -= b.x4();
    a.x5() -= b.x5();
    a.x6() -= b.x6();
}


//mandel6x1 addition
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator+(
    mandel6x1<T> const& a, 
    mandel6x1<U> const& b)
{
  return mandel6x1<decltype(a.x1()*b.x1())>(
    a.x1() + b.x1(),
    a.x2() + b.x2(),
    a.x3() + b.x3(),
    a.x4() + b.x4(),
    a.x5() + b.x5(),
    a.x6() + b.x6(),
    false);
}

//mandel6x1 += addition
template <class T>
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
void operator+=(
    mandel6x1<T>& a, 
    mandel6x1<T> const& b)
{
    a.x1() += b.x1();
    a.x2() += b.x2();
    a.x3() += b.x3();
    a.x4() += b.x4();
    a.x5() += b.x5();
    a.x6() += b.x6();
}

//mandel6x1 subtraction
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator-(
    mandel6x1<T> const& a, 
    mandel6x1<U> const& b)
{
  return mandel6x1<decltype(a.x1()+b.x1())>(
    a.x1() - b.x1(),
    a.x2() - b.x2(),
    a.x3() - b.x3(),
    a.x4() - b.x4(),
    a.x5() - b.x5(),
    a.x6() - b.x6(),
    false);
}

//mandel6x1 negation
template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
mandel6x1<T> operator-(
    mandel6x1<T> const& a)
{
  return mandel6x1<T>(
    -a.x1(),
    -a.x2(),
    -a.x3(),
    -a.x4(),
    -a.x5(),
    -a.x6(),
    false);
}

/***************************************************************************** 
 * Linear Algebra for Mandel6x1 (2nd order tensor)
 *****************************************************************************/
//trace
template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
T trace(
    mandel6x1<T> const& a)
{
  return a.x1() + a.x2() + a.x3();
}

//determinate
template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
T determinant(
    mandel6x1<T> const& t)
{
      return t.x1()    * (t.x2()*t.x3()    - t.x4()*t.x4()/t.two) -
             t.x6()/t.r2 * (t.x3()*t.x6()/t.r2 - t.x4()*t.x5()/t.two) +
             t.x5()/t.r2 * (t.x6()*t.x4()/t.two - t.x2()*t.x5()/t.r2);
}

//inverse
template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
mandel6x1<T> inverse(
    mandel6x1<T> const& V)
{
    //Direct calculation of inverse of (6x1) 2nd-order Mandel tensor 
    T det = determinant(V);

    return mandel6x1<T>(
            (V.x2()*V.x3()    - V.x4()*V.x4()/V.two)/det,
            (V.x1()*V.x3()    - V.x5()*V.x5()/V.two)/det,
            (V.x1()*V.x2()    - V.x6()*V.x6()/V.two)/det,
            (V.x6()*V.x5()/V.two - V.x1()*V.x4()/V.r2)/det,
            (V.x6()*V.x4()/V.two - V.x2()*V.x5()/V.r2)/det,
            (V.x4()*V.x5()/V.two - V.x6()*V.x3()/V.r2)/det,
            true);
    //not in mandel form anymore; return to mandel form for consistency with 
    //other functions
}

/** Tensor multiply MandelVector (6x1) by MandelVector (6x1) **/
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(
    mandel6x1<T> const &v,
    mandel6x1<U> const &t)
{
    mandel6x1<T> tt = t;
    mandel6x1<U> vv = v;
    tt.invMandelXform();
    vv.invMandelXform();

    using result_type = decltype(v.xx() * t.xx());
    return mandel6x1<result_type>(
          tt.x1()*vv.x1()+tt.x6()*vv.x6()+tt.x5()*vv.x5(),
          tt.x6()*vv.x6()+tt.x2()*vv.x2()+tt.x4()*vv.x4(),
          tt.x5()*vv.x5()+tt.x4()*vv.x4()+tt.x3()*vv.x3(),
          tt.x6()*vv.x5()+tt.x2()*vv.x4()+tt.x4()*vv.x3(),
          tt.x1()*vv.x5()+tt.x6()*vv.x4()+tt.x5()*vv.x3(),
          tt.x1()*vv.x6()+tt.x6()*vv.x2()+tt.x5()*vv.x4(),
          true);
}

/** Tensor multiply MandelVector (6x1) by symmetric3x3 (3x3) **/
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(
    mandel6x1<T> const &v, 
    symmetric3x3<U> const &tt)
{
    mandel6x1<T> vv = v;
    vv.invMandelXform();

    using result_type = decltype(tt.xx() * v.xx());
    return mandel6x1<result_type>(
          tt.xx()*vv.x1()+tt.xy()*vv.x6()+tt.xz()*vv.x5(),
          tt.xy()*vv.x6()+tt.yy()*vv.x2()+tt.yz()*vv.x4(),
          tt.xz()*vv.x5()+tt.yz()*vv.x4()+tt.zz()*vv.x3(),
          tt.xy()*vv.x5()+tt.yy()*vv.x4()+tt.yz()*vv.x3(),
          tt.xx()*vv.x5()+tt.xy()*vv.x4()+tt.xz()*vv.x3(),
          tt.xx()*vv.x6()+tt.xy()*vv.x2()+tt.xz()*vv.x4(),
          true);
}

/** Tensor multiply symmetric3x3 by MandelVector (6x1) **/
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(
    symmetric3x3<T> const &tt,
    mandel6x1<U> const &v) 
{
    mandel6x1<U> vv = v;
    vv.invMandelXform();

    using result_type = decltype(tt.xx() * v.xx());
    return mandel6x1<result_type>(
          tt.xx()*vv.x1()+tt.xy()*vv.x6()+tt.xz()*vv.x5(),
          tt.xy()*vv.x6()+tt.yy()*vv.x2()+tt.yz()*vv.x4(),
          tt.xz()*vv.x5()+tt.yz()*vv.x4()+tt.zz()*vv.x3(),
          tt.xy()*vv.x5()+tt.yy()*vv.x4()+tt.yz()*vv.x3(),
          tt.xx()*vv.x5()+tt.xy()*vv.x4()+tt.xz()*vv.x3(),
          tt.xx()*vv.x6()+tt.xy()*vv.x2()+tt.xz()*vv.x4(),
          true);
}

/** Tensor multiply MandelVector (6x1) by diagonal3x3 (3x3) **/
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(mandel6x1<T> const &v, diagonal3x3<U> const &tt)
{
    mandel6x1<T> vv = v;
    vv.invMandelXform();

    using result_type = decltype(tt.xx() * v.xx());
    return mandel6x1<result_type>(
          tt.xx()*vv.x1(),
          tt.yy()*vv.x2(),
          tt.zz()*vv.x3(),
          tt.yy()*vv.x4(),
          tt.xx()*vv.x5(),
          tt.xx()*vv.x6(),
          true);
}

/** Tensor multiply diagonal3x3 (3x3) by MandelVector (6x1) **/
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(diagonal3x3<T> const &tt, mandel6x1<U> const &v)
{
    mandel6x1<U> vv = v;
    vv.invMandelXform();

    using result_type = decltype(tt.xx() * v.xx());
    return mandel6x1<result_type>(
          tt.xx()*vv.x1(),
          tt.yy()*vv.x2(),
          tt.zz()*vv.x3(),
          tt.yy()*vv.x4(),
          tt.xx()*vv.x5(),
          tt.xx()*vv.x6(),
          true);
}

/** Tensor multiply MandelVector (6x1) by vector3 (3x1) **/
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(mandel6x1<T> const &m, vector3<U> const &v)
{
  mandel6x1<T> t = m;
  t.invMandelXform();
  //The transformation is reversed as the operation is performed 
  // by first converting to a tensor.
  using result_type = decltype(t.xx() * v.x());
  return vector3<result_type>(
          t.x1()*v.x() + t.x6()*v.y() + t.x5()*v.z(),
          t.x6()*v.x() + t.x2()*v.y() + t.x4()*v.z(),
          t.x5()*v.x() + t.x4()*v.y() + t.x3()*v.z());
}

//partial pivoting in <dynamic_matrix.hpp>

//////////////////////////////////////////////////////////////////////////////////
// Operations Yielding Scalars (double dot product of two tensors (2nd order)
//////////////////////////////////////////////////////////////////////////////////
//double dot product of mandel6x1 with mandel6x1
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto ddot(mandel6x1<T> const &t, mandel6x1<U> const &v)
{
    using result_type = decltype(t.x1() * v.x());
    return result_type(t.x1()*v.x1() + t.x2()*v.x2() + t.x3()*v.x3() + t.two*t.x4()*v.x4() + t.two*t.x5()*v.x5() + t.two*t.x6()*v.x6());
}

//double dot product of mandel6x1 with diagonal3x3
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto ddot(mandel6x1<T> const &t, diagonal3x3<U> const &d)
{
    using result_type = decltype(t.x1() * d.x());
    return result_type(t.x1()*d.xx() + t.x2()*d.yy() + t.x3()*d.zz());
}

//double dot product of diagonal3x3 and mandel6x1
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto ddot(diagonal3x3<T> const &d, mandel6x1<U> const &t)
{
    using result_type = decltype(d.xx() * t.x1());
    return result_type(t.x1()*d.xx() + t.x2()*d.yy() + t.x3()*d.zz());
}

//double dot product of mandel6x1 and symmetric3x3
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto ddot( mandel6x1<T> const &v, symmetric3x3<U> const &t)
{
    using result_type = decltype(v.x1() * t.xx());
    return result_type(t.xx()*v.x1() + t.yy()*v.x2() + t.zz()*v.x3() + v.two*t.yz()*v.x4() + v.two*t.xz()*v.x5() + v.two*t.xy()*v.x6());
}

//double dot product of symmetric3x3 and mandel6x1
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto ddot(symmetric3x3<T> const& t, mandel6x1<U> const &v) 
{
    using result_type = decltype(v.x1() * t.xx());
    return result_type(t.xx()*v.x1() + t.yy()*v.x2() + t.zz()*v.x3() + v.two*t.yz()*v.x4() + v.two*t.xz()*v.x5() + v.two*t.xy()*v.x6());
}

//double dot product of mandel6x1 and matrix3x3
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto ddot(mandel6x1<T> const& v, matrix3x3<U> const& t)
{
    using result_type = decltype(t.xx() * v.x1());
    return result_type(t.xx()*v.x1() + t.yy()*v.x2() + t.zz()*v.x3() + t.yz()*v.x4() + t.xz()*v.x5() + t.xy()*v.x6() + t.zy()*v.x4() + t.zx()*v.x5() + t.yx()*v.x6());
}

//double dot product of matrix3x3 and mandel6x1 
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto ddot(matrix3x3<T> const& t, mandel6x1<U> const& v)
{
    using result_type = decltype(v.x1() * t.xx());
    return result_type(t.xx()*v.x1() + t.yy()*v.x2() + t.zz()*v.x3() + t.yz()*v.x4() + t.xz()*v.x5() + t.xy()*v.x6() + t.zy()*v.x4() + t.zx()*v.x5() + t.yx()*v.x6());
}
//////////////////////////////////////////////////////////////////////////////////
// Operations Yielding matrix3x3
//////////////////////////////////////////////////////////////////////////////////
/** Tensor mult: mandel6x1 by matrix3x3 **/ 
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(mandel6x1<T> const& v, matrix3x3<U> const& t)
{ 
    using result_type = decltype(t.xx() * v.x1());
    //The transformation is reversed as the operation is performed 
    // by first converting to a tensor.
    return matrix3x3<result_type>(v.x1()*t.xx()+v.x6()/v.r2*t.yx()+v.x5()/v.r2*t.zx(),v.x1()*t.xy()+v.x6()/v.r2*t.yy()+v.x5()/v.r2*t.zy(),v.x1()*t.xz()+v.x6()/v.r2*t.yz()+v.x5()/v.r2*t.zz(),
                       v.x6()/v.r2*t.xx()+v.x2()*t.yx()+v.x4()/v.r2*t.zx(),v.x6()/v.r2*t.xy()+v.x2()*t.yy()+v.x4()/v.r2*t.zy(),v.x6()/v.r2*t.xz()+v.x2()*t.yz()+v.x4()/v.r2*t.zz(),
                       v.x5()/v.r2*t.xx()+v.x4()/v.r2*t.yx()+v.x3()*t.zx(),v.x5()/v.r2*t.xy()+v.x4()/v.r2*t.yy()+v.x3()*t.zy(),v.x5()/v.r2*t.xz()+v.x4()/v.r2*t.yz()+v.x3()*t.zz());
}

/** Tensor mult: matrix3x3 by mandel6x1 **/ 
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(matrix3x3<T> const& t, mandel6x1<U> const& v)
{ 
    //The transformation is reversed as the operation is performed 
    // by first converting to a tensor.
    using result_type = decltype(t.xx() * v.x1());
    return matrix3x3<result_type>(t.xx()*v.x1()+t.xy()*v.x6()/v.r2+t.xz()*v.x5()/v.r2,t.xx()*v.x6()/v.r2+t.xy()*v.x2()+t.xz()*v.x4()/v.r2,t.xx()*v.x5()/v.r2+t.xy()*v.x4()/v.r2+t.xz()*v.x3(),
                  t.yx()*v.x1()+t.yy()*v.x6()/v.r2+t.yz()*v.x5()/v.r2,t.yx()*v.x6()/v.r2+t.yy()*v.x2()+t.yz()*v.x4()/v.r2,t.yx()*v.x5()/v.r2+t.yy()*v.x4()/v.r2+t.yz()*v.x3(),
                  t.zx()*v.x1()+t.zy()*v.x6()/v.r2+t.zz()*v.x5()/v.r2,t.zx()*v.x6()/v.r2+t.zy()*v.x2()+t.zz()*v.x4()/v.r2,t.zx()*v.x5()/v.r2+t.zy()*v.x4()/v.r2+t.zz()*v.x3());
}

//////////////////////////////////////////////////////////////////////////////////
// Operations Yielding symmetric3x3 from a mandel6x1 
//////////////////////////////////////////////////////////////////////////////////

/** Convert MandelVector (6x1) to symmetric3x3 **/
template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
symmetric3x3<T> mandel6x1_to_symmetric3x3(mandel6x1<T> const& v)
{
    //invert Mandel Tranformation of MandelVector 
    return symmetric3x3<T>(v.x1(),v.x6()/v.r2,v.x5()/v.r2,
                                  v.x2()   ,v.x4()/v.r2,
                                            v.x3());
}

/** Convert MandelVector (6x1) to symmetric3x3 **/
template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
matrix3x3<T> mandel6x1_to_matrix3x3(mandel6x1<T> const& v)
{
    //invert Mandel Tranformation of MandelVector 
    return matrix3x3<T>(v.x1(),     v.x6()/v.r2,v.x5()/v.r2,
                        v.x6()/v.r2,v.x2()     ,v.x4()/v.r2,
                        v.x5()/v.r2,v.x4()/v.r2,v.x3());
}

//misc
inline int constexpr mandel6x1_component_count = 6;

//output print
template <class U>
P3A_ALWAYS_INLINE constexpr 
std::ostream& operator<<(std::ostream& os, mandel6x1<U> const& a)
{
  os << std::cout.precision(4);
  os << std::scientific;
  os << "\t  | " << a.x1()       << " " << a.x6()*a.r2i << " " << a.x5()*a.r2i << " |" <<std::endl;
  os << "\t  | " << a.x6()*a.r2i << " " << a.x2()       << " " << a.x4()*a.r2i << " |" <<std::endl;
  os << "\t  | " << a.x5()*a.r2i << " " << a.x4()*a.r2i << " " << a.x3()       << " |" <<std::endl;

  return os;
}

}
