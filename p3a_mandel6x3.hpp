#pragma once

#include "p3a_macros.hpp"
#include "p3a_diagonal3x3.hpp"
#include "p3a_vector3.hpp"
#include "p3a_symmetric3x3.hpp"
#include "p3a_matrix3x3.hpp"
#include "p3a_mandel6x1.hpp"
#include "p3a_mandel6x6.hpp"
#include "p3a_mandel3x6.hpp"
/********************************* NOTES ************************************
 * This header provides the class: 
 *
 * - `mandel6x3` (6x3) 3rd order Tensor
 *
 *   Constructors:
 *   
 *   - `mandel6x3(mandel6x3)`
 *   - `mandel6x3(<list of values>)`
 *   - `mandel6x3(static_matrix<6,3>)` -- includes testing for symmetry
 *
 * See additional notes in `p3a_mandel6x1.hpp`.
 */

namespace p3a {

/******************************************************************/
/******************************************************************/
template <class T>
class Mandel6x3
/** 
 * Represents a 3th order tensor as a 6x3 Mandel array
 */
/******************************************************************/
{
 T x11,x12,x13,
   x21,x22,x23,
   x31,x32,x33,
   x41,x42,x43,
   x51,x52,x53,
   x61,x62,x63;
 bool applyTransform;

 template<class T>
 P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr static
 T r2() {std::sqrt(T(2.0));}

 template<class T>
 P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr static
 T r2i() {T(1.0)/std::sqrt(T(2.0));}

 template<class T>
 P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr static 
 T two() {T(2.0);}

 public:
  /**** constructors, destructors, and assigns ****/
  P3A_ALWAYS_INLINE constexpr
  mandel6x3() = default;

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  mandel6x6(
      T const& X11, T const& X12, T const& X13,
      T const& X21, T const& X22, T const& X23,
      T const& X31, T const& X32, T const& X33,
      T const& X41, T const& X42, T const& X43,
      T const& X51, T const& X52, T const& X53,
      T const& X61, T const& X62, T const& X63):
      x11(X11),x12(X12),x13(X13),
      x21(X21),x22(X22),x23(X23),
      x31(X31),x32(X32),x33(X33),
      x41(X41),x42(X42),x43(X43),
      x51(X51),x52(X52),x53(X53),
      x61(X61),x62(X62),x63(X63),
       applyTransform(true)
  {
    this->MandelXform();
  }

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  mandel6x6(
      T const& X11, T const& X12, T const& X13,
      T const& X21, T const& X22, T const& X23,
      T const& X31, T const& X32, T const& X33,
      T const& X41, T const& X42, T const& X43,
      T const& X51, T const& X52, T const& X53,
      T const& X61, T const& X62, T const& X63, 
      bool const& Xform):
    x11(X11),x12(X12),x13(X13),
    x21(X21),x22(X22),x23(X23),
    x31(X31),x32(X32),x33(X33),
    x41(X41),x42(X42),x43(X43),
    x51(X51),x52(X52),x53(X53),
    x61(X61),x62(X62),x63(X63),
    applyTransform(Xform)
  {
    if (applyTransform)
        this->MandelXform();
  }
  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  mandel6x3(
      static_matrix<T,6,3> const& X, bool const& Xform):
    x11(X(0,0)),x12(X(0,1)),x13(X(0,2)),
    x21(X(1,0)),x22(X(1,1)),x23(X(1,2)),
    x31(X(2,0)),x32(X(2,1)),x33(X(2,2)),
    x41(X(3,0)),x42(X(3,1)),x43(X(3,2)),
    x51(X(4,0)),x52(X(4,1)),x53(X(4,2)),
    x61(X(5,0)),x62(X(5,1)),x63(X(5,2)),
    applyTransform(Xform)
  {
    if (applyTransform)
        this->MandelXform();
  }

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  mandel6x3(
      static_matrix<T,6,3> const& X):
    x11(X(0,0)),x12(X(0,1)),x13(X(0,2)),
    x21(X(1,0)),x22(X(1,1)),x23(X(1,2)),
    x31(X(2,0)),x32(X(2,1)),x33(X(2,2)),
    x41(X(3,0)),x42(X(3,1)),x43(X(3,2)),
    x51(X(4,0)),x52(X(4,1)),x53(X(4,2)),
    x61(X(5,0)),x62(X(5,1)),x63(X(5,2)),
    applyTransform(true)
  {
    this->MandelXform();
  }

  //return by mandel index 1-6
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x11() const { return x11; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x12() const { return x12; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x13() const { return x13; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x21() const { return x21; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x22() const { return x22; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x23() const { return x23; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x31() const { return x31; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x32() const { return x32; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x33() const { return x33; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x41() const { return x41; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x42() const { return x42; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x43() const { return x43; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x51() const { return x51; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x52() const { return x52; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x53() const { return x53; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x61() const { return x61; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x62() const { return x62; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x63() const { return x63; }

  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x11() { return x11; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x12() { return x12; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x13() { return x13; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x21() { return x21; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x22() { return x22; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x23() { return x23; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x31() { return x31; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x32() { return x32; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x33() { return x33; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x41() { return x41; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x42() { return x42; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x43() { return x43; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x51() { return x51; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x52() { return x52; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x53() { return x53; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x61() { return x61; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x62() { return x62; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x63() { return x63; }

  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE static constexpr
  mandel6x3<T> zero()
  {
    return mandel6x3<T>(
        T(0), T(0), T(0),
        T(0), T(0), T(0),
        T(0), T(0), T(0),
        T(0), T(0), T(0),
        T(0), T(0), T(0),
        T(0), T(0), T(0),
        false);
  }

  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE static constexpr
  mandel6x3<T> identity()
  {
    return mandel6x3<T>(
        T(1), T(0), T(0),
        T(0), T(1), T(0),
        T(0), T(0), T(1),
        T(0), T(0), T(0),
        T(0), T(0), T(0),
        T(0), T(0), T(0),
        true);
  }

  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  void MandelXform()
  {
       x41*=r2i, x42*=r2i, x43*=r2i,
       x51*=r2i, x52*=r2i, x53*=r2i,
       x61*=r2i, x62*=r2i, x63*=r2i;
  }

  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  void invMandelXform()
  {
       x41/=r2i, x42/=r2i, x43/=r2i,
       x51/=r2i, x52/=r2i, x53/=r2i,
       x61/=r2i, x62/=r2i, x63/=r2i;
  }

};

/***************************************************************************** 
 * Operator overloads for mandel6x3 tensors (3rd order tensor)
 *****************************************************************************/
//mandel6x3 binary operators with scalars
//multiplication by constant
template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<B>, mandel6x6<decltype(A() * B())>>::type
operator*(
        mandel6x3<A> const& a, 
        B const& c)
{
    return mandel6x3<decltype(a.x11()*c)>(
            a.x11()*c, a.x12()*c, a.x13()*c,
            a.x21()*c, a.x22()*c, a.x23()*c,
            a.x31()*c, a.x32()*c, a.x33()*c,
            a.x41()*c, a.x42()*c, a.x43()*c,
            a.x51()*c, a.x52()*c, a.x53()*c,
            a.x61()*c, a.x62()*c, a.x63()*c,
            false);
}

//multiplication by constant
template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<A>, mandel6x6<decltype(A() * B())>>::type 
operator*(
        A const& c, 
        mandel6x3<B> const& t)
{
    return t * c;
}

//division by constant
template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<B>, mandel6x6<decltype(A() * B())>>::type
operator/(
        mandel6x3<A> const& a, 
        B const& c)
{
    return mandel6x3<decltype<(a.x11() / c)>(
            a.x11()/c, a.x12()/c, a.x13()/c,
            a.x21()/c, a.x22()/c, a.x23()/c,
            a.x31()/c, a.x32()/c, a.x33()/c,
            a.x41()/c, a.x42()/c, a.x43()/c,
            a.x51()/c, a.x52()/c, a.x53()/c,
            a.x61()/c, a.x62()/c, a.x63()/c,
            false);
}

//division /= by constant
template <class A>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
void operator/=(
        mandel6x3<A>& a, 
        A const& c)
{
        a.x11()/=c; 
        a.x12()/=c; 
        a.x13()/=c; 
        a.x21()/=c;
        a.x22()/=c;
        a.x23()/=c;
        a.x31()/=c;
        a.x32()/=c;
        a.x33()/=c;
        a.x41()/=c;
        a.x42()/=c;
        a.x43()/=c;
        a.x51()/=c;
        a.x52()/=c;
        a.x53()/=c;
        a.x61()/=c;
        a.x62()/=c;
        a.x63()/=c;
}

//multiplication *= by constant
template <class A>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
void operator*=(
        mandel6x3<A>& a, 
        A const& c)
{
        a.x11()*=c; 
        a.x12()*=c; 
        a.x13()*=c; 
        a.x21()*=c;
        a.x22()*=c;
        a.x23()*=c;
        a.x31()*=c;
        a.x32()*=c;
        a.x33()*=c;
        a.x41()*=c;
        a.x42()*=c;
        a.x43()*=c;
        a.x51()*=c;
        a.x52()*=c;
        a.x53()*=c;
        a.x61()*=c;
        a.x62()*=c;
        a.x63()*=c;
}

//mandel6x6 += addition
template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
void operator+=(
    mandel6x6<T>& a, 
    mandel6x6<T> const& b)
{
    a.x11() += b.x11();
    a.x12() += b.x12();
    a.x13() += b.x13();
    a.x21() += b.x21();
    a.x22() += b.x22();
    a.x23() += b.x23(); 
    a.x31() += b.x31();
    a.x32() += b.x32();
    a.x33() += b.x33(); 
    a.x41() += b.x41();
    a.x42() += b.x42();
    a.x43() += b.x43();
    a.x51() += b.x51();
    a.x52() += b.x52();
    a.x53() += b.x53();
    a.x61() += b.x61(); 
    a.x62() += b.x62();
    a.x63() += b.x63();
}

//mandel6x3 -= subtraction
template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
void operator-=(
    mandel6x3<T>& a, 
    mandel6x3<T> const& b)
{
    a.x11() -= b.x11();
    a.x12() -= b.x12();
    a.x13() -= b.x13();
    a.x21() -= b.x21();
    a.x22() -= b.x22();
    a.x23() -= b.x23(); 
    a.x31() -= b.x31();
    a.x32() -= b.x32();
    a.x33() -= b.x33(); 
    a.x41() -= b.x41();
    a.x42() -= b.x42();
    a.x43() -= b.x43();
    a.x51() -= b.x51();
    a.x52() -= b.x52();
    a.x53() -= b.x53();
    a.x61() -= b.x61(); 
    a.x62() -= b.x62();
    a.x63() -= b.x63();
}

//mandel6x3 addition
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator+(
    mandel6x3<T> const& a, 
    mandel6x3<U> const& b)
{
  return mandel6x3<decltype(a.x11()+b.x11())>(
    a.x11() + b.x11(), a.x12() + b.x12(), a.x13() + b.x13(),
    a.x21() + b.x21(), a.x22() + b.x22(), a.x23() + b.x23(),
    a.x31() + b.x31(), a.x32() + b.x32(), a.x33() + b.x33(),
    a.x41() + b.x41(), a.x42() + b.x42(), a.x43() + b.x43(),
    a.x51() + b.x51(), a.x52() + b.x52(), a.x53() + b.x53(),
    a.x61() + b.x61(), a.x62() + b.x62(), a.x63() + b.x63(),
    false);
}

//mandel6x3 subtraction
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator-(
    mandel6x3<T> const& a, 
    mandel6x3<U> const& b)
{
  return mandel6x3<decltype(a.x11()-b.x11())>(
    a.x11() - b.x11(), a.x12() - b.x12(), a.x13() - b.x13(), 
    a.x21() - b.x21(), a.x22() - b.x22(), a.x23() - b.x23(),
    a.x31() - b.x31(), a.x32() - b.x32(), a.x33() - b.x33(),
    a.x41() - b.x41(), a.x42() - b.x42(), a.x43() - b.x43(),
    a.x51() - b.x51(), a.x52() - b.x52(), a.x53() - b.x53(),
    a.x61() - b.x61(), a.x62() - b.x62(), a.x63() - b.x63(),
    false);
}

//mandel6x3 negation
template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
mandel6x3<T> operator-(
    mandel6x3<T> const& a)
{
  return mandel6x3<T>(
    -a.x11(), -a.x12(), -a.x13(),
    -a.x21(), -a.x22(), -a.x23(),
    -a.x31(), -a.x32(), -a.x33(),
    -a.x41(), -a.x42(), -a.x43(),
    -a.x51(), -a.x52(), -a.x53(),
    -a.x61(), -a.x62(), -a.x63(),
    false);
}

/***************************************************************************** 
 * Linear Algebra for mandel6x3 (3rd order tensor)
 *****************************************************************************/
//trace
template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
T trace(
    mandel6x3<T> const& a)
{
  return a.x11() + a.x22() + a.x33();
}

/** transpose 6x3 to 3x6 **/
template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
mandel3x6<T> transpose(
    mandel6x3<T> const &d)
{
    return mandel3x6<T>(
        d.x11(), d.x21(), d.x31(), d.x41(), d.x51(), d.x61(),
        d.x12(), d.x22(), d.x32(), d.x42(), d.x52(), d.x62(),
        d.x13(), d.x23(), d.x33(), d.x43(), d.x53(), d.x63(),
        false); //already transformed
}

/** transpose 3x6 to 6x3 **/
template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
mandel6x3<T> transpose(
    mandel3x6<T> const &d)
{
    return mandel6x3<T>(
        d.x11(), d.x21(), d.x31(),
        d.x12(), d.x22(), d.x32(),
        d.x13(), d.x23(), d.x33(),
        d.x14(), d.x24(), d.x34(),
        d.x15(), d.x25(), d.x35(),
        d.x16(), d.x26(), d.x36(),
        false); //already transformed
}

/** Tensor multiply mandel3x6 (3x6) by mandel6x3 (6x3) **/
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(
    mandel3x6<T> const &e,
    mandel6x3<U> const &d)
{
    return matrix3x3<decltype(e.x11*d.x11)>(
            e.x11()*d.x11() + e.x12()*d.x21() + e.x13()*d.x31() + e.x14()*d.x41() + e.x15()*d.x51() + e.x16()*d.x61(),
            e.x11()*d.x12() + e.x12()*d.x22() + e.x13()*d.x32() + e.x14()*d.x42() + e.x15()*d.x52() + e.x16()*d.x62(),
            e.x11()*d.x13() + e.x12()*d.x23() + e.x13()*d.x33() + e.x14()*d.x43() + e.x15()*d.x53() + e.x16()*d.x63(),
            e.x21()*d.x11() + e.x22()*d.x21() + e.x23()*d.x31() + e.x24()*d.x41() + e.x25()*d.x51() + e.x26()*d.x61(),
            e.x21()*d.x12() + e.x22()*d.x22() + e.x23()*d.x32() + e.x24()*d.x42() + e.x25()*d.x52() + e.x26()*d.x62(),
            e.x21()*d.x13() + e.x22()*d.x23() + e.x23()*d.x33() + e.x24()*d.x43() + e.x25()*d.x53() + e.x26()*d.x63(),
            e.x31()*d.x11() + e.x32()*d.x21() + e.x33()*d.x31() + e.x34()*d.x41() + e.x35()*d.x51() + e.x36()*d.x61(),
            e.x31()*d.x12() + e.x32()*d.x22() + e.x33()*d.x32() + e.x34()*d.x42() + e.x35()*d.x52() + e.x36()*d.x62(),
            e.x31()*d.x13() + e.x32()*d.x23() + e.x33()*d.x33() + e.x34()*d.x43() + e.x35()*d.x53() + e.x36()*d.x63());
}

/** Tensor multiply Mandel6x3 (6x3) by matrix3x3 (3x3) **/
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(
    mandel6x3<T> const &e, 
    matrix3x3<U> const &d)
{
    return mandel6x3<decltype(e.x11()*d.xx())>(
            e.x11()*d.xx() + e.x12()*d.yx() + e.x13()*d.zx(), e.x11()*d.xy() + e.x12()*d.yy() + e.x13()*d.zy(), e.x11()*d.xz() + e.x12()*d.yz() + e.x13()*d.zz(),
            e.x21()*d.xx() + e.x22()*d.yx() + e.x23()*d.zx(), e.x21()*d.xy() + e.x22()*d.yy() + e.x23()*d.zy(), e.x21()*d.xz() + e.x22()*d.yz() + e.x23()*d.zz(),
            e.x31()*d.xx() + e.x32()*d.yx() + e.x33()*d.zx(), e.x31()*d.xy() + e.x32()*d.yy() + e.x33()*d.zy(), e.x31()*d.xz() + e.x32()*d.yz() + e.x33()*d.zz(),
            e.x41()*d.xx() + e.x42()*d.yx() + e.x43()*d.zx(), e.x41()*d.xy() + e.x42()*d.yy() + e.x43()*d.zy(), e.x41()*d.xz() + e.x42()*d.yz() + e.x43()*d.zz(),
            e.x51()*d.xx() + e.x52()*d.yx() + e.x53()*d.zx(), e.x51()*d.xy() + e.x52()*d.yy() + e.x53()*d.zy(), e.x51()*d.xz() + e.x52()*d.yz() + e.x53()*d.zz(),
            e.x61()*d.xx() + e.x62()*d.yx() + e.x63()*d.zx(), e.x61()*d.xy() + e.x62()*d.yy() + e.x63()*d.zy(), e.x61()*d.xz() + e.x62()*d.yz() + e.x63()*d.zz(),
            false);//already transformed
}

/** Tensor Dot mandel6x3 (6x3) by mandel6x1 (6x1) **/
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(
    mandel6x3<T> const &e, 
    mandel6x1<U> const &f)
{
    using result_type decltype(e.x11(),f.x11()) 
    matrix3x3<result_type> d = f;
    return mandel6x3<result_type>(
            e.x11()*d.xx() + e.x12()*d.yx() + e.x13()*d.zx(), e.x11()*d.xy() + e.x12()*d.yy() + e.x13()*d.zy(), e.x11()*d.xz() + e.x12()*d.yz() + e.x13()*d.zz(),
            e.x21()*d.xx() + e.x22()*d.yx() + e.x23()*d.zx(), e.x21()*d.xy() + e.x22()*d.yy() + e.x23()*d.zy(), e.x21()*d.xz() + e.x22()*d.yz() + e.x23()*d.zz(),
            e.x31()*d.xx() + e.x32()*d.yx() + e.x33()*d.zx(), e.x31()*d.xy() + e.x32()*d.yy() + e.x33()*d.zy(), e.x31()*d.xz() + e.x32()*d.yz() + e.x33()*d.zz(),
            e.x41()*d.xx() + e.x42()*d.yx() + e.x43()*d.zx(), e.x41()*d.xy() + e.x42()*d.yy() + e.x43()*d.zy(), e.x41()*d.xz() + e.x42()*d.yz() + e.x43()*d.zz(),
            e.x51()*d.xx() + e.x52()*d.yx() + e.x53()*d.zx(), e.x51()*d.xy() + e.x52()*d.yy() + e.x53()*d.zy(), e.x51()*d.xz() + e.x52()*d.yz() + e.x53()*d.zz(),
            e.x61()*d.xx() + e.x62()*d.yx() + e.x63()*d.zx(), e.x61()*d.xy() + e.x62()*d.yy() + e.x63()*d.zy(), e.x61()*d.xz() + e.x62()*d.yz() + e.x63()*d.zz(),
            false);//already transformed
}

/** Tensor Dot mandel6x3 (6x3) by symmetric3x3 (3x3) **/
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(
    mandel6x3<T> const &e, 
    symmetric3x3<U> const &f)
{
    using result_type decltype(e.x11(),f.x11()) 
    matrix3x3<result_type> d = f;
    return mandel6x3<result_type>(
            e.x11()*d.xx() + e.x12()*d.yx() + e.x13()*d.zx(), e.x11()*d.xy() + e.x12()*d.yy() + e.x13()*d.zy(), e.x11()*d.xz() + e.x12()*d.yz() + e.x13()*d.zz(),
            e.x21()*d.xx() + e.x22()*d.yx() + e.x23()*d.zx(), e.x21()*d.xy() + e.x22()*d.yy() + e.x23()*d.zy(), e.x21()*d.xz() + e.x22()*d.yz() + e.x23()*d.zz(),
            e.x31()*d.xx() + e.x32()*d.yx() + e.x33()*d.zx(), e.x31()*d.xy() + e.x32()*d.yy() + e.x33()*d.zy(), e.x31()*d.xz() + e.x32()*d.yz() + e.x33()*d.zz(),
            e.x41()*d.xx() + e.x42()*d.yx() + e.x43()*d.zx(), e.x41()*d.xy() + e.x42()*d.yy() + e.x43()*d.zy(), e.x41()*d.xz() + e.x42()*d.yz() + e.x43()*d.zz(),
            e.x51()*d.xx() + e.x52()*d.yx() + e.x53()*d.zx(), e.x51()*d.xy() + e.x52()*d.yy() + e.x53()*d.zy(), e.x51()*d.xz() + e.x52()*d.yz() + e.x53()*d.zz(),
            e.x61()*d.xx() + e.x62()*d.yx() + e.x63()*d.zx(), e.x61()*d.xy() + e.x62()*d.yy() + e.x63()*d.zy(), e.x61()*d.xz() + e.x62()*d.yz() + e.x63()*d.zz(),
            false);//already transformed
}

/** Tensor multiply MandelVector (6x3) by diagonal3x3 (3x3) **/
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(
    mandel6x6<T> const &v, 
    diagonal3x3<U> const &d)
{
    return mandel6x3<decltype(e.x11(),d.xx())>(
            e.x11()*d.xx(), e.x12()*d.yy(), e.x13()*d.zz(),
            e.x21()*d.xx(), e.x22()*d.yy(), e.x23()*d.zz(),
            e.x31()*d.xx(), e.x32()*d.yy(), e.x33()*d.zz(),
            e.x41()*d.xx(), e.x42()*d.yy(), e.x43()*d.zz(),
            e.x51()*d.xx(), e.x52()*d.yy(), e.x53()*d.zz(),
            e.x61()*d.xx(), e.x62()*d.yy(), e.x63()*d.zz(),
            false);//already transformed
}

/** Tensor multiply MandelVector (6x3) by vector3 (3x1) **/
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(
    mandel6x3<T> const &C, 
    vector3<U> const &v)
{
    return mandel6x1<decltype(C.x11()*v.x())>(
            C.x11()*v.x() + C.x12()*v.y() + C.x13()*v.z(),
            C.x21()*v.x() + C.x22()*v.y() + C.x23()*v.z(),
            C.x31()*v.x() + C.x32()*v.y() + C.x33()*v.z(),
            C.x41()*v.x() + C.x42()*v.y() + C.x43()*v.z(),
            C.x51()*v.x() + C.x52()*v.y() + C.x53()*v.z(),
            C.x61()*v.x() + C.x62()*v.y() + C.x63()*v.z(),
            false);//already transformed
}

//misc
inline int constexpr mandel6x3_component_count = 18;

