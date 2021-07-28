#pragma once

#include "p3a_macros.hpp"
#include "p3a_diagonal3x3.hpp"
#include "p3a_vector3.hpp"
#include "p3a_symmetric3x3.hpp"
#include "p3a_matrix3x3.hpp"
#include "p3a_mandel6x1.hpp"
#include "p3a_mandel6x6.hpp"
/********************************* NOTES ************************************
 * This header provides the class: 
 *
 * - `mandel3x6` (3x6) 3rd order Tensor
 *
 *   Constructors:
 *   
 *   - `mandel3x6(mandel3x6)`
 *   - `mandel3x6(<list of values>)`
 *   - `mandel3x6(static_matrix<3,6>)` -- includes testing for symmetry
 *
 * See additional notes in `p3a_mandel6x1.hpp`.
 */

namespace p3a {

/******************************************************************/
/******************************************************************/
template <class T>
class Mandel3x6
/** 
 * Represents a 3rd order tensor as a 3x6 Mandel array
 */
/******************************************************************/
{
 T x11,x12,x13,x14,x15,x16,
   x21,x22,x23,x24,x25,x26,
   x31,x32,x33,x34,x35,x36;
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
  mandel3x6() = default;

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  mandel3x6(
      T const& X11, T const& X12, T const& X13, T const& X14, T const& X15, T const& X16,
      T const& X21, T const& X22, T const& X23, T const& X24, T const& X25, T const& X26,
      T const& X31, T const& X32, T const& X33, T const& X34, T const& X35, T const& X36):
      x11(X11),x12(X12),x13(X13),x14(X14),x15(X15),x16(X16),
      x21(X21),x22(X22),x23(X23),x24(X24),x25(X25),x26(X26),
      x31(X31),x32(X32),x33(X33),x34(X34),x35(X35),x36(X36),
      applyTransform(true)
  {
    if (applyTransform)
        this->MandelXform();
  }

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  mandel3x6(
    T const& X11, T const& X12, T const& X13, T const& X14, T const& X15, T const& X16,
    T const& X21, T const& X22, T const& X23, T const& X24, T const& X25, T const& X26,
    T const& X31, T const& X32, T const& X33, T const& X34, T const& X35, T const& X36,
    bool const& Xform):
    x11(X11),x12(X12),x13(X13),x14(X14),x15(X15),x16(X16),
    x21(X21),x22(X22),x23(X23),x24(X24),x25(X25),x26(X26),
    x31(X31),x32(X32),x33(X33),x34(X34),x35(X35),x36(X36),
    applyTransform(Xform)
  {
    if (applyTransform)
        this->MandelXform();
  }

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  mandel3x6(
      static_matrix<T,3,6> const& X, bool const& Xform):
    x11(X(0,0)),x12(X(0,1)),x13(X(0,2)),x14(X(0,3)),x15(X(0,4)),x16(X(0,5)),
    x21(X(1,0)),x22(X(1,1)),x23(X(1,2)),x24(X(1,3)),x25(X(1,4)),x26(X(1,5)),
    x31(X(2,0)),x32(X(2,1)),x33(X(2,2)),x34(X(2,3)),x35(X(2,4)),x36(X(2,5)),
    applyTransform(Xform)
  {
    if (applyTransform)
        this->MandelXform();
  }

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  mandel3x6(
      static_matrix<T,3,6> const& X):
    x11(X(0,0)),x12(X(0,1)),x13(X(0,2)),x14(X(0,3)),x15(X(0,4)),x16(X(0,5)),
    x21(X(1,0)),x22(X(1,1)),x23(X(1,2)),x24(X(1,3)),x25(X(1,4)),x26(X(1,5)),
    x31(X(2,0)),x32(X(2,1)),x33(X(2,2)),x34(X(2,3)),x35(X(2,4)),x36(X(2,5)),
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
  T const& x14() const { return x14; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x15() const { return x15; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x16() const { return x16; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x21() const { return x21; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x22() const { return x22; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x23() const { return x23; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x24() const { return x24; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x25() const { return x25; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x26() const { return x26; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x31() const { return x31; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x32() const { return x32; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x33() const { return x33; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x34() const { return x34; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x35() const { return x35; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x36() const { return x36; }

  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x11() { return x11; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x12() { return x12; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x13() { return x13; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x14() { return x14; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x15() { return x15; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x16() { return x16; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x21() { return x21; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x22() { return x22; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x23() { return x23; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x24() { return x24; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x25() { return x25; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x26() { return x26; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x31() { return x31; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x32() { return x32; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x33() { return x33; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x34() { return x34; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x35() { return x35; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x36() { return x36; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr

  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE static constexpr
  mandel3x6<T> zero()
  {
    return mandel3x6<T>(
        T(0), T(0), T(0), T(0), T(0), T(0),
        T(0), T(0), T(0), T(0), T(0), T(0),
        T(0), T(0), T(0), T(0), T(0), T(0),
        false);
  }

  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE static constexpr
  mandel6x6<T> identity()
  {
    return mandel3x6<T>(
        T(1), T(0), T(0), T(0), T(0), T(0),
        T(0), T(1), T(0), T(0), T(0), T(0),
        T(0), T(0), T(1), T(0), T(0), T(0),
        true);
  }

  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  void MandelXform()
  {
        x14*=r2i, x15*=r2i, x16*=r2i, 
        x24*=r2i, x25*=r2i, x26*=r2i,
        x34*=r2i, x35*=r2i, x36*=r2i;
  }

  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  void invMandelXform()
  {
        x14/=r2i, x15/=r2i, x16/=r2i, 
        x24/=r2i, x25/=r2i, x26/=r2i,
        x34/=r2i, x35/=r2i, x36/=r2i;
  }

};

/***************************************************************************** 
 * Operator overloads for mandel3x6 tensors (3rd order tensor)
 *****************************************************************************/
//mandel3x6 binary operators with scalars
//multiplication by constant
template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<B>, mandel3x6<decltype(A() * B())>>::type
operator*(
        mandel3x6<A> const& a, 
        B const& c)
{
    return mandel3x6<decltype(a.x11()*c)>(
            a.x11()*c, a.x12()*c, a.x13()*c, a.x14()*c, a.x15()*c, a.x16()*c,
            a.x21()*c, a.x22()*c, a.x23()*c, a.x24()*c, a.x25()*c, a.x26()*c,
            a.x31()*c, a.x32()*c, a.x33()*c, a.x34()*c, a.x35()*c, a.x36()*c,
            false);
}

//multiplication by constant
template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<A>, mandel3x6<decltype(A() * B())>>::type 
operator*(
        A const& c, 
        mandel3x6<B> const& t)
{
    return t * c;
}

//division by constant
template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<B>, mandel3x6<decltype(A() * B())>>::type
operator/(
        mandel3x6<A> const& a, 
        B const& c)
{
    return mandel6x6<decltype<(a.x11() / c)>(
            a.x11()/c, a.x12()/c, a.x13()/c, a.x14()/c, a.x15()/c, a.x16()/c,
            a.x21()/c, a.x22()/c, a.x23()/c, a.x24()/c, a.x25()/c, a.x26()/c,
            a.x31()/c, a.x32()/c, a.x33()/c, a.x34()/c, a.x35()/c, a.x36()/c,
            false);
}

//division /= by constant
template <class A>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
void operator/=(
        mandel3x6<A>& a, 
        A const& c)
{
        a.x11()/=c; 
        a.x12()/=c; 
        a.x13()/=c; 
        a.x14()/=c; 
        a.x15()/=c;
        a.x16()/=c;
        a.x21()/=c;
        a.x22()/=c;
        a.x23()/=c;
        a.x24()/=c;
        a.x25()/=c;
        a.x26()/=c;
        a.x31()/=c;
        a.x32()/=c;
        a.x33()/=c;
        a.x34()/=c;
        a.x35()/=c;
        a.x36()/=c;
}

//multiplication *= by constant
template <class A>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
void operator*=(
        mandel3x6<A>& a, 
        A const& c)
{
        a.x11()*=c; 
        a.x12()*=c; 
        a.x13()*=c; 
        a.x14()*=c; 
        a.x15()*=c;
        a.x16()*=c;
        a.x21()*=c;
        a.x22()*=c;
        a.x23()*=c;
        a.x24()*=c;
        a.x25()*=c;
        a.x26()*=c;
        a.x31()*=c;
        a.x32()*=c;
        a.x33()*=c;
        a.x34()*=c;
        a.x35()*=c;
        a.x36()*=c;
}

//mandel3x6 += addition
template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
void operator+=(
    mandel3x6<T>& a, 
    mandel3x6<T> const& b)
{
    a.x11() += b.x11();
    a.x12() += b.x12();
    a.x13() += b.x13();
    a.x14() += b.x14();
    a.x15() += b.x15();
    a.x16() += b.x16();
    a.x21() += b.x21();
    a.x22() += b.x22();
    a.x23() += b.x23(); 
    a.x24() += b.x24();
    a.x25() += b.x25();
    a.x26() += b.x26();
    a.x31() += b.x31();
    a.x32() += b.x32();
    a.x33() += b.x33(); 
    a.x34() += b.x34();
    a.x35() += b.x35();
    a.x36() += b.x36();
}

//mandel3x6 -= subtraction
template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
void operator-=(
    mandel3x6<T>& a, 
    mandel3x6<T> const& b)
{
    a.x11() -= b.x11();
    a.x12() -= b.x12();
    a.x13() -= b.x13();
    a.x14() -= b.x14();
    a.x15() -= b.x15();
    a.x16() -= b.x16();
    a.x21() -= b.x21();
    a.x22() -= b.x22();
    a.x23() -= b.x23(); 
    a.x24() -= b.x24();
    a.x25() -= b.x25();
    a.x26() -= b.x26();
    a.x31() -= b.x31();
    a.x32() -= b.x32();
    a.x33() -= b.x33(); 
    a.x34() -= b.x34();
    a.x35() -= b.x35();
    a.x36() -= b.x36();
}

//mandel3x6 addition
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator+(
    mandel3x6<T> const& a, 
    mandel3x6<U> const& b)
{
  return mandel3x6<decltype(a.x11()+b.x11())>(
    a.x11() + b.x11(), a.x12() + b.x12(), a.x13() + b.x13(), a.x14() + b.x14(), a.x15() + b.x15(), a.x16() + b.x16(),
    a.x21() + b.x21(), a.x22() + b.x22(), a.x23() + b.x23(), a.x24() + b.x24(), a.x25() + b.x25(), a.x26() + b.x26(),
    a.x31() + b.x31(), a.x32() + b.x32(), a.x33() + b.x33(), a.x34() + b.x34(), a.x35() + b.x35(), a.x36() + b.x36(),
    false);
}

//mandel3x6 subtraction
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator-(
    mandel3x6<T> const& a, 
    mandel3x6<U> const& b)
{
  return mandel3x6<decltype(a.x11()-b.x11())>(
    a.x11() - b.x11(), a.x12() - b.x12(), a.x13() - b.x13(), a.x14() - b.x14(), a.x15() - b.x15(), a.x16() - b.x16(),
    a.x21() - b.x21(), a.x22() - b.x22(), a.x23() - b.x23(), a.x24() - b.x24(), a.x25() - b.x25(), a.x26() - b.x26(),
    a.x31() - b.x31(), a.x32() - b.x32(), a.x33() - b.x33(), a.x34() - b.x34(), a.x35() - b.x35(), a.x36() - b.x36(),
    false);
}

//mandel3x6 negation
template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
mandel3x6<T> operator-(
    mandel3x6<T> const& a)
{
  return mandel3x6<T>(
    -a.x11(), -a.x12(), -a.x13(), -a.x14(), -a.x15(), -a.x16(),
    -a.x21(), -a.x22(), -a.x23(), -a.x24(), -a.x25(), -a.x26(),
    -a.x31(), -a.x32(), -a.x33(), -a.x34(), -a.x35(), -a.x36(),
    false);
}

/***************************************************************************** 
 * Linear Algebra for mandel3x6 (3rd order tensor)
 *****************************************************************************/
//trace
template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
T trace(
    mandel3x6<T> const& a)
{
  return a.x11() + a.x22() + a.x33();
}

/** Tensor multiply mandel3x6 (3x6) by mandel6x1 (6x1) **/
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(
    mandel3x6<T> const &C,
    mandel6x1<U> const &v)
{
    return vector3<decltype(C.x11()*v.x1())>(
        (C.x11()*v.x1() + C.x12()*v.x2() + C.x13()*v.x3() + C.x14()*v.x4() + C.x15()*v.x5() + C.x16()*v.x6()),
        (C.x21()*v.x1() + C.x22()*v.x2() + C.x23()*v.x3() + C.x24()*v.x4() + C.x25()*v.x5() + C.x26()*v.x6()),
        (C.x31()*v.x1() + C.x32()*v.x2() + C.x33()*v.x3() + C.x34()*v.x4() + C.x35()*v.x5() + C.x36()*v.x6()))
}

/** Tensor multiply mandel3x6 (3x6) by mandel6x6 (6x6) **/
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(
    mandel3x6<T> const &e,
    mandel6x6<U> const &k)
{
    return mandel3x6<decltype(e.x11()*k.x11())>(
            e.x11()*k.x11() + e.x12()*k.x21() + e.x13()*k.x31() + e.x14()*k.x41() + e.x15()*k.x51() + e.x16()*k.x61(),
            e.x11()*k.x12() + e.x12()*k.x22() + e.x13()*k.x32() + e.x14()*k.x42() + e.x15()*k.x52() + e.x16()*k.x62(),
            e.x11()*k.x13() + e.x12()*k.x23() + e.x13()*k.x33() + e.x14()*k.x43() + e.x15()*k.x53() + e.x16()*k.x63(),
            e.x11()*k.x14() + e.x12()*k.x24() + e.x13()*k.x34() + e.x14()*k.x44() + e.x15()*k.x54() + e.x16()*k.x64(),
            e.x11()*k.x15() + e.x12()*k.x25() + e.x13()*k.x35() + e.x14()*k.x45() + e.x15()*k.x55() + e.x16()*k.x65(),
            e.x11()*k.x16() + e.x12()*k.x26() + e.x13()*k.x36() + e.x14()*k.x46() + e.x15()*k.x56() + e.x16()*k.x66(),
            e.x21()*k.x11() + e.x22()*k.x21() + e.x23()*k.x31() + e.x24()*k.x41() + e.x25()*k.x51() + e.x26()*k.x61(),
            e.x21()*k.x12() + e.x22()*k.x22() + e.x23()*k.x32() + e.x24()*k.x42() + e.x25()*k.x52() + e.x26()*k.x62(),
            e.x21()*k.x13() + e.x22()*k.x23() + e.x23()*k.x33() + e.x24()*k.x43() + e.x25()*k.x53() + e.x26()*k.x63(),
            e.x21()*k.x14() + e.x22()*k.x24() + e.x23()*k.x34() + e.x24()*k.x44() + e.x25()*k.x54() + e.x26()*k.x64(),
            e.x21()*k.x15() + e.x22()*k.x25() + e.x23()*k.x35() + e.x24()*k.x45() + e.x25()*k.x55() + e.x26()*k.x65(),
            e.x21()*k.x16() + e.x22()*k.x26() + e.x23()*k.x36() + e.x24()*k.x46() + e.x25()*k.x56() + e.x26()*k.x66(),
            e.x31()*k.x11() + e.x32()*k.x21() + e.x33()*k.x31() + e.x34()*k.x41() + e.x35()*k.x51() + e.x36()*k.x61(),
            e.x31()*k.x12() + e.x32()*k.x22() + e.x33()*k.x32() + e.x34()*k.x42() + e.x35()*k.x52() + e.x36()*k.x62(),
            e.x31()*k.x13() + e.x32()*k.x23() + e.x33()*k.x33() + e.x34()*k.x43() + e.x35()*k.x53() + e.x36()*k.x63(),
            e.x31()*k.x14() + e.x32()*k.x24() + e.x33()*k.x34() + e.x34()*k.x44() + e.x35()*k.x54() + e.x36()*k.x64(),
            e.x31()*k.x15() + e.x32()*k.x25() + e.x33()*k.x35() + e.x34()*k.x45() + e.x35()*k.x55() + e.x36()*k.x65(),
            e.x31()*k.x16() + e.x32()*k.x26() + e.x33()*k.x36() + e.x34()*k.x46() + e.x35()*k.x56() + e.x36()*k.x66(),
            false);//already transformed
}

/** Tensor multiply mandel3x6 (3x6) by mandel6x6 (6x6) **/
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(
    mandel3x6<T> const &e,
    mandel6x6<U> const &k)
{
    return mandel3x6<decltype(e.x11()*k.x11())>(
            e.x11()*k.x11() + e.x12()*k.x21() + e.x13()*k.x31() + e.x14()*k.x41() + e.x15()*k.x51() + e.x16()*k.x61(),
            e.x11()*k.x12() + e.x12()*k.x22() + e.x13()*k.x32() + e.x14()*k.x42() + e.x15()*k.x52() + e.x16()*k.x62(),
            e.x11()*k.x13() + e.x12()*k.x23() + e.x13()*k.x33() + e.x14()*k.x43() + e.x15()*k.x53() + e.x16()*k.x63(),
            e.x11()*k.x14() + e.x12()*k.x24() + e.x13()*k.x34() + e.x14()*k.x44() + e.x15()*k.x54() + e.x16()*k.x64(),
            e.x11()*k.x15() + e.x12()*k.x25() + e.x13()*k.x35() + e.x14()*k.x45() + e.x15()*k.x55() + e.x16()*k.x65(),
            e.x11()*k.x16() + e.x12()*k.x26() + e.x13()*k.x36() + e.x14()*k.x46() + e.x15()*k.x56() + e.x16()*k.x66(),
            e.x21()*k.x11() + e.x22()*k.x21() + e.x23()*k.x31() + e.x24()*k.x41() + e.x25()*k.x51() + e.x26()*k.x61(),
            e.x21()*k.x12() + e.x22()*k.x22() + e.x23()*k.x32() + e.x24()*k.x42() + e.x25()*k.x52() + e.x26()*k.x62(),
            e.x21()*k.x13() + e.x22()*k.x23() + e.x23()*k.x33() + e.x24()*k.x43() + e.x25()*k.x53() + e.x26()*k.x63(),
            e.x21()*k.x14() + e.x22()*k.x24() + e.x23()*k.x34() + e.x24()*k.x44() + e.x25()*k.x54() + e.x26()*k.x64(),
            e.x21()*k.x15() + e.x22()*k.x25() + e.x23()*k.x35() + e.x24()*k.x45() + e.x25()*k.x55() + e.x26()*k.x65(),
            e.x21()*k.x16() + e.x22()*k.x26() + e.x23()*k.x36() + e.x24()*k.x46() + e.x25()*k.x56() + e.x26()*k.x66(),
            e.x31()*k.x11() + e.x32()*k.x21() + e.x33()*k.x31() + e.x34()*k.x41() + e.x35()*k.x51() + e.x36()*k.x61(),
            e.x31()*k.x12() + e.x32()*k.x22() + e.x33()*k.x32() + e.x34()*k.x42() + e.x35()*k.x52() + e.x36()*k.x62(),
            e.x31()*k.x13() + e.x32()*k.x23() + e.x33()*k.x33() + e.x34()*k.x43() + e.x35()*k.x53() + e.x36()*k.x63(),
            e.x31()*k.x14() + e.x32()*k.x24() + e.x33()*k.x34() + e.x34()*k.x44() + e.x35()*k.x54() + e.x36()*k.x64(),
            e.x31()*k.x15() + e.x32()*k.x25() + e.x33()*k.x35() + e.x34()*k.x45() + e.x35()*k.x55() + e.x36()*k.x65(),
            e.x31()*k.x16() + e.x32()*k.x26() + e.x33()*k.x36() + e.x34()*k.x46() + e.x35()*k.x56() + e.x36()*k.x66(),
            false);//already transformed
}

/** Tensor multiply symmetric3x3 (3x3) by mandel3x6 (3x6) **/
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(
    symmetric3x3<U> const &s)
    mandel3x6<T> const &C, 
{
    v = mandel6x1<T>(s);
    return C*v;
}

/** Tensor multiply symmetric3x3 by MandelVector (6x1) **/
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(
    symmetric3x3<T> const &tt,
    mandel3x6<U> const &v
    )
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
auto operator*(diagonal3x3<T> const &tt, mandel6x1<U> const &vv)
{
    using result_type = decltype(tt.xx() * vv.xx());
    return mandel6x1<result_type>(
          tt.xx()*vv.x1(),
          tt.yy()*vv.x2(),
          tt.zz()*vv.x3(),
          tt.yy()*vv.x4(),
          tt.xx()*vv.x5(),
          tt.xx()*vv.x6(),
          false);
}

//misc
inline int constexpr mandel3x6_component_count = 18;
