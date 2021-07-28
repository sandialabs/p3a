#pragma once

#include "p3a_macros.hpp"
#include "p3a_diagonal3x3.hpp"
#include "p3a_vector3.hpp"
#include "p3a_symmetric3x3.hpp"
#include "p3a_matrix3x3.hpp"
#include "p3a_mandel6x1.hpp"
#include "p3a_static_vector.hpp"
#include "p3a_static_matrix.hpp"
/********************************* NOTES ************************************
 * This header provides the class: 
 *
 * - `mandel6x6` (6x6) 4th order Tensor
 *   
 *   Constructors:
 *   
 *   - `mandel6x6(mandel6x6)`
 *   - `mandel6x6(<list of values>)`
 *   - `mandel6x6(static_matrix<6,6>)`
 *
 * See additional notes in `p3a_mandel6x1.hpp`.
 */

namespace p3a {

/******************************************************************/
/******************************************************************/
template <class T>
class Mandel6x6
/** 
 * Represents a 4th order tensor as a 6x6 Mandel array
 */
/******************************************************************/
{
 T x11,x12,x13,x14,x15,x16,
   x21,x22,x23,x24,x25,x26,
   x31,x32,x33,x34,x35,x36,
   x41,x42,x43,x44,x45,x46,
   x51,x52,x53,x54,x55,x56,
   x61,x62,x63,x64,x65,x66;
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
  mandel6x6() = default;

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  mandel6x6(
      T const& X11, T const& X12, T const& X13, T const& X14, T const& X15, T const& X16,
      T const& X21, T const& X22, T const& X23, T const& X24, T const& X25, T const& X26,
      T const& X31, T const& X32, T const& X33, T const& X34, T const& X35, T const& X36,
      T const& X41, T const& X42, T const& X43, T const& X44, T const& X45, T const& X46,
      T const& X51, T const& X52, T const& X53, T const& X54, T const& X55, T const& X56,
      T const& X61, T const& X62, T const& X63, T const& X64, T const& X65, T const& X66):
      x11(X11),x12(X12),x13(X13),x14(X14),x15(X15),x16(X16),
      x21(X21),x22(X22),x23(X23),x24(X24),x25(X25),x26(X26),
      x31(X31),x32(X32),x33(X33),x34(X34),x35(X35),x36(X36),
      x41(X41),x42(X42),x43(X43),x44(X44),x45(X45),x46(X46),
      x51(X51),x52(X52),x53(X53),x54(X54),x55(X55),x56(X56),
      x61(X61),x62(X62),x63(X63),x64(X64),x65(X65),x66(X66),
       applyTransform(true)
  {
    if (applyTransform)
        this->MandelXform();
  }

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  mandel6x6(
      T const& X11, T const& X12, T const& X13, T const& X14, T const& X15, T const& X16,
      T const& X21, T const& X22, T const& X23, T const& X24, T const& X25, T const& X26,
      T const& X31, T const& X32, T const& X33, T const& X34, T const& X35, T const& X36,
      T const& X41, T const& X42, T const& X43, T const& X44, T const& X45, T const& X46,
      T const& X51, T const& X52, T const& X53, T const& X54, T const& X55, T const& X56,
      T const& X61, T const& X62, T const& X63, T const& X64, T const& X65, T const& X66, 
      bool const& Xform):
    x11(X11),x12(X12),x13(X13),x14(X14),x15(X15),x16(X16),
    x21(X21),x22(X22),x23(X23),x24(X24),x25(X25),x26(X26),
    x31(X31),x32(X32),x33(X33),x34(X34),x35(X35),x36(X36),
    x41(X41),x42(X42),x43(X43),x44(X44),x45(X45),x46(X46),
    x51(X51),x52(X52),x53(X53),x54(X54),x55(X55),x56(X56),
    x61(X61),x62(X62),x63(X63),x64(X64),x65(X65),x66(X66),
    applyTransform(Xform)
  {
    if (applyTransform)
        this->MandelXform();
  }

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  mandel6x6(
      static_matrix<T,6,6> const& X, bool const& Xform):
    x11(X(0,0)),x12(X(0,1)),x13(X(0,2)),x14(X(0,3)),x15(X(0,4)),x16(X(0,5)),
    x21(X(1,0)),x22(X(1,1)),x23(X(1,2)),x24(X(1,3)),x25(X(1,4)),x26(X(1,5)),
    x31(X(2,0)),x32(X(2,1)),x33(X(2,2)),x34(X(2,3)),x35(X(2,4)),x36(X(2,5)),
    x41(X(3,0)),x42(X(3,1)),x43(X(3,2)),x44(X(3,3)),x45(X(3,4)),x46(X(3,5)),
    x51(X(4,0)),x52(X(4,1)),x53(X(4,2)),x54(X(4,3)),x55(X(4,4)),x56(X(4,5)),
    x61(X(5,0)),x62(X(5,1)),x63(X(5,2)),x64(X(5,3)),x65(X(5,4)),x66(X(5,5)),
    applyTransform(Xform)
  {
    if (applyTransform)
        this->MandelXform();
  }

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  mandel6x6(
      static_matrix<T,6,6> const& X):
    x11(X(0,0)),x12(X(0,1)),x13(X(0,2)),x14(X(0,3)),x15(X(0,4)),x16(X(0,5)),
    x21(X(1,0)),x22(X(1,1)),x23(X(1,2)),x24(X(1,3)),x25(X(1,4)),x26(X(1,5)),
    x31(X(2,0)),x32(X(2,1)),x33(X(2,2)),x34(X(2,3)),x35(X(2,4)),x36(X(2,5)),
    x41(X(3,0)),x42(X(3,1)),x43(X(3,2)),x44(X(3,3)),x45(X(3,4)),x46(X(3,5)),
    x51(X(4,0)),x52(X(4,1)),x53(X(4,2)),x54(X(4,3)),x55(X(4,4)),x56(X(4,5)),
    x61(X(5,0)),x62(X(5,1)),x63(X(5,2)),x64(X(5,3)),x65(X(5,4)),x66(X(5,5)),
    applyTransform(true)
  {
    this->MandelXform();
  }

  //return by mandel index 1-6,1-6
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
  T const& x41() const { return x41; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x42() const { return x42; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x43() const { return x43; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x44() const { return x44; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x45() const { return x45; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x46() const { return x46; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x51() const { return x51; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x52() const { return x52; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x53() const { return x53; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x54() const { return x54; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x55() const { return x55; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x56() const { return x56; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x61() const { return x61; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x62() const { return x62; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x63() const { return x63; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x64() const { return x64; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x65() const { return x65; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x66() const { return x66; }

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
  T& x41() { return x41; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x42() { return x42; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x43() { return x43; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x44() { return x44; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x45() { return x45; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x46() { return x46; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x51() { return x51; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x52() { return x52; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x53() { return x53; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x54() { return x54; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x55() { return x55; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x56() { return x56; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x61() { return x61; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x62() { return x62; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x63() { return x63; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x64() { return x64; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x65() { return x65; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x66() { return x66; }

  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE static constexpr
  mandel6x6<T> zero()
  {
    return mandel6x6<T>(
        T(0), T(0), T(0), T(0), T(0), T(0),
        T(0), T(0), T(0), T(0), T(0), T(0),
        T(0), T(0), T(0), T(0), T(0), T(0),
        T(0), T(0), T(0), T(0), T(0), T(0),
        T(0), T(0), T(0), T(0), T(0), T(0),
        T(0), T(0), T(0), T(0), T(0), T(0),
        false);
  }

  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE static constexpr
  mandel6x6<T> identity()
  {
    return mandel6x6<T>(
        T(1), T(0), T(0), T(0), T(0), T(0),
        T(0), T(1), T(0), T(0), T(0), T(0),
        T(0), T(0), T(1), T(0), T(0), T(0),
        T(0), T(0), T(0), T(1), T(0), T(0),
        T(0), T(0), T(0), T(0), T(1), T(0),
        T(0), T(0), T(0), T(0), T(0), T(1),
        true);
  }

  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  void MandelXform()
  {
                              x14*=r2,x15*=r2,x16*=r2;
                              x24*=r2,x25*=r2,x26*=r2;
                              x34*=r2,x35*=r2,x36*=r2;
      x41*=r2,x42*=r2,x43*=r2,x44*=two,x45*=two,x46*=two;
      x51*=r2,x52*=r2,x53*=r2,x54*=two,x55*=two,x56*=two;
      x61*=r2,x62*=r2,x63*=r2,x64*=two,x65*=two,x66*=two;
  }

  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  void invMandelXform()
  {
                              x14/=r2,x15/=r2,x16/=r2;
                              x24/=r2,x25/=r2,x26/=r2;
                              x34/=r2,x35/=r2,x36/=r2;
      x41/=r2,x42/=r2,x43/=r2,x44/=two,x45/=two,x46/=two;
      x51/=r2,x52/=r2,x53/=r2,x54/=two,x55/=two,x56/=two;
      x61/=r2,x62/=r2,x63/=r2,x64/=two,x65/=two,x66/=two;
  }

};

/***************************************************************************** 
 * Operator overloads for mandel6x6 tensors (4th order tensor)
 *****************************************************************************/
//mandel6x6 binary operators with scalars
//multiplication by constant
template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<B>, mandel6x6<decltype(A() * B())>>::type
operator*(
        mandel6x6<A> const& a, 
        B const& c)
{
    return mandel6x6<decltype(a.x11()*c)>(
            a.x11()*c, a.x12()*c, a.x13()*c, a.x14()*c, a.x15()*c, a.x16()*c,
            a.x21()*c, a.x22()*c, a.x23()*c, a.x24()*c, a.x25()*c, a.x26()*c,
            a.x31()*c, a.x32()*c, a.x33()*c, a.x34()*c, a.x35()*c, a.x36()*c,
            a.x41()*c, a.x42()*c, a.x43()*c, a.x44()*c, a.x45()*c, a.x46()*c,
            a.x51()*c, a.x52()*c, a.x53()*c, a.x54()*c, a.x55()*c, a.x56()*c,
            a.x61()*c, a.x62()*c, a.x63()*c, a.x64()*c, a.x65()*c, a.x66()*c,
            false);
}

//multiplication by constant
template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<A>, mandel6x6<decltype(A() * B())>>::type 
operator*(
        A const& c, 
        mandel6x6<B> const& t)
{
    return t * c;
}

//division by constant
template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<B>, mandel6x6<decltype(A() * B())>>::type
operator/(
        mandel6x6<A> const& a, 
        B const& c)
{
    return mandel6x6<decltype<(a.x11() / c)>(
            a.x11()/c, a.x12()/c, a.x13()/c, a.x14()/c, a.x15()/c, a.x16()/c,
            a.x21()/c, a.x22()/c, a.x23()/c, a.x24()/c, a.x25()/c, a.x26()/c,
            a.x31()/c, a.x32()/c, a.x33()/c, a.x34()/c, a.x35()/c, a.x36()/c,
            a.x41()/c, a.x42()/c, a.x43()/c, a.x44()/c, a.x45()/c, a.x46()/c,
            a.x51()/c, a.x52()/c, a.x53()/c, a.x54()/c, a.x55()/c, a.x56()/c,
            a.x61()/c, a.x62()/c, a.x63()/c, a.x64()/c, a.x65()/c, a.x66()/c,
            false);
}

//division /= by constant
template <class A>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
void operator/=(
        mandel6x6<A>& a, 
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
        a.x41()/=c;
        a.x42()/=c;
        a.x43()/=c;
        a.x44()/=c;
        a.x45()/=c;
        a.x46()/=c;
        a.x51()/=c;
        a.x52()/=c;
        a.x53()/=c;
        a.x54()/=c;
        a.x55()/=c;
        a.x56()/=c;
        a.x61()/=c;
        a.x62()/=c;
        a.x63()/=c;
        a.x64()/=c;
        a.x65()/=c;
        a.x66()/=c;
}

//multiplication *= by constant
template <class A>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
void operator*=(
        mandel6x6<A>& a, 
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
        a.x41()*=c;
        a.x42()*=c;
        a.x43()*=c;
        a.x44()*=c;
        a.x45()*=c;
        a.x46()*=c;
        a.x51()*=c;
        a.x52()*=c;
        a.x53()*=c;
        a.x54()*=c;
        a.x55()*=c;
        a.x56()*=c;
        a.x61()*=c;
        a.x62()*=c;
        a.x63()*=c;
        a.x64()*=c;
        a.x65()*=c;
        a.x66()*=c;
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
    a.x41() += b.x41();
    a.x42() += b.x42();
    a.x43() += b.x43();
    a.x44() += b.x44();
    a.x45() += b.x45();
    a.x46() += b.x46();
    a.x51() += b.x51();
    a.x52() += b.x52();
    a.x53() += b.x53();
    a.x54() += b.x54();
    a.x55() += b.x55();
    a.x56() += b.x56();
    a.x61() += b.x61(); 
    a.x62() += b.x62();
    a.x63() += b.x63();
    a.x64() += b.x64();
    a.x65() += b.x65();
    a.x66() += b.x66();
}

//mandel6x6 -= subtraction
template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
void operator-=(
    mandel6x6<T>& a, 
    mandel6x6<T> const& b)
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
    a.x41() -= b.x41();
    a.x42() -= b.x42();
    a.x43() -= b.x43();
    a.x44() -= b.x44();
    a.x45() -= b.x45();
    a.x46() -= b.x46();
    a.x51() -= b.x51();
    a.x52() -= b.x52();
    a.x53() -= b.x53();
    a.x54() -= b.x54();
    a.x55() -= b.x55();
    a.x56() -= b.x56();
    a.x61() -= b.x61(); 
    a.x62() -= b.x62();
    a.x63() -= b.x63();
    a.x64() -= b.x64();
    a.x65() -= b.x65();
    a.x66() -= b.x66();
}
//mandel6x6 addition
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator+(
    mandel6x6<T> const& a, 
    mandel6x6<U> const& b)
{
  return mandel6x6<decltype(a.x11()+b.x11())>(
    a.x11() + b.x11(), a.x12() + b.x12(), a.x13() + b.x13(), a.x14() + b.x14(), a.x15() + b.x15(), a.x16() + b.x16(),
    a.x21() + b.x21(), a.x22() + b.x22(), a.x23() + b.x23(), a.x24() + b.x24(), a.x25() + b.x25(), a.x26() + b.x26(),
    a.x31() + b.x31(), a.x32() + b.x32(), a.x33() + b.x33(), a.x34() + b.x34(), a.x35() + b.x35(), a.x36() + b.x36(),
    a.x41() + b.x41(), a.x42() + b.x42(), a.x43() + b.x43(), a.x44() + b.x44(), a.x45() + b.x45(), a.x46() + b.x46(),
    a.x51() + b.x51(), a.x52() + b.x52(), a.x53() + b.x53(), a.x54() + b.x54(), a.x55() + b.x55(), a.x56() + b.x56(),
    a.x61() + b.x61(), a.x62() + b.x62(), a.x63() + b.x63(), a.x64() + b.x64(), a.x65() + b.x65(), a.x66() + b.x66(),
    false);
}

//mandel6x6 subtraction
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator-(
    mandel6x6<T> const& a, 
    mandel6x6<U> const& b)
{
  return mandel6x6<decltype(a.x11()-b.x11())>(
    a.x11() - b.x11(), a.x12() - b.x12(), a.x13() - b.x13(), a.x14() - b.x14(), a.x15() - b.x15(), a.x16() - b.x16(),
    a.x21() - b.x21(), a.x22() - b.x22(), a.x23() - b.x23(), a.x24() - b.x24(), a.x25() - b.x25(), a.x26() - b.x26(),
    a.x31() - b.x31(), a.x32() - b.x32(), a.x33() - b.x33(), a.x34() - b.x34(), a.x35() - b.x35(), a.x36() - b.x36(),
    a.x41() - b.x41(), a.x42() - b.x42(), a.x43() - b.x43(), a.x44() - b.x44(), a.x45() - b.x45(), a.x46() - b.x46(),
    a.x51() - b.x51(), a.x52() - b.x52(), a.x53() - b.x53(), a.x54() - b.x54(), a.x55() - b.x55(), a.x56() - b.x56(),
    a.x61() - b.x61(), a.x62() - b.x62(), a.x63() - b.x63(), a.x64() - b.x64(), a.x65() - b.x65(), a.x66() - b.x66(),
    false);
}

//mandel6x1 negation
template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
mandel6x6<T> operator-(
    mandel6x6<T> const& a)
{
  return mandel6x6<T>(
    -a.x11(), -a.x12(), -a.x13(), -a.x14(), -a.x15(), -a.x16(),
    -a.x21(), -a.x22(), -a.x23(), -a.x24(), -a.x25(), -a.x26(),
    -a.x31(), -a.x32(), -a.x33(), -a.x34(), -a.x35(), -a.x36(),
    -a.x41(), -a.x42(), -a.x43(), -a.x44(), -a.x45(), -a.x46(),
    -a.x51(), -a.x52(), -a.x53(), -a.x54(), -a.x55(), -a.x56(),
    -a.x61(), -a.x62(), -a.x63(), -a.x64(), -a.x65(), -a.x66(),
    false);
}

/***************************************************************************** 
 * Linear Algebra for Mandel6x6(4th order tensor)
 *****************************************************************************/
//trace
template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
T trace(
    mandel6x6<T> const& a)
{
  return a.x11() + a.x22() + a.x33() + a.x44() + a.x55() + a.x66();
}

//inverse
template <class U>
P3A_NEVER_INLINE 
mandel6x6<U> Inverse(
    mandel6x6<U> const &V)
{
    /**************************************************************************
    * This procedure computes the inverse of a MandelTensor (6x6) using
    * Gauss-Jordan elimination with partial pivoting.
    *
    * Converted to C++ from F90/95 and uses MandelTensor
    * class. Algorithm taken from "alegranevada/alegra/electromech/tensor_toolkit.f90".
    ***************************************************************************/
    mandel6x6<U> T(V);
    T.invMandelXform();
    //......................................................................local
    int row=0;
    int col=0;
    int icond = 0;
    int startrow = -1;
    int v = 0;
    int n = 6 ;
    int span = 5;
    int loc = 0;
    U vval=0;
    U wmax =0;
    U fac =0; 
    U wcond = minimum<U>();
    static_matrix<U,6,6> w;
    static_matrix<U,6,6> ainv;
    static_vector<U,6> dummy;

     
    dummy.zeros();
    ainv.zeros();
    ainv.identity();
    w.zeros();

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Initialize and allocate
    //
    // Make working tensor from input Mandel Tensor
    // while converting input 6x6 tensor T from Mandel form to normal form
    
    // Row 1
    w(0,0)=T.x11;
    w(0,1)=T.x12;
    w(0,2)=T.x13;
    w(0,3)=T.x14;
    w(0,4)=T.x15;
    w(0,5)=T.x16;
    // Row 2
    w(1,0)=T.x21;
    w(1,1)=T.x22;
    w(1,2)=T.x23;
    w(1,3)=T.x24;
    w(1,4)=T.x25;
    w(1,5)=T.x26;
    //Row 3
    w(2,0)=T.x31;
    w(2,1)=T.x32;
    w(2,2)=T.x33;
    w(2,3)=T.x34;
    w(2,4)=T.x35;
    w(2,5)=T.x36;
    //Row 4
    w(3,0)=T.x41;
    w(3,1)=T.x42;
    w(3,2)=T.x43;
    w(3,3)=T.x44;
    w(3,4)=T.x45;
    w(3,5)=T.x46;
    //Row 5
    w(4,0)=T.x51;
    w(4,1)=T.x52;
    w(4,2)=T.x53;
    w(4,3)=T.x54;
    w(4,4)=T.x55;
    w(4,5)=T.x56;
    //Row 6
    w(5,0)=T.x61;
    w(5,1)=T.x62;
    w(5,2)=T.x63;
    w(5,3)=T.x64;
    w(5,4)=T.x65;
    w(5,5)=T.x66;

    //Locate maximum value in each row and normalize by that value
    for(row=0;row<n;row++)
    {
        wmax = 0.0;
        for(col=0;col<n;col++)
        {
            //find max value of each row
            if(std::abs(w(row,col))>std::abs(wmax))
            {
                wmax = w(row,col);
            }
        }
        //if the row is empty: error
        if(std::abs(wmax) <= minimum<U>();
        {
           icond = 1;
           throw std::invalid_argument(
                   "p3a_mandel6x6::Inverse: ERROR, row is zero during inversion of 4th order Mandel Tensor");
        }
        //normalize rows by max value in row
        for(col=0;col<n;col++)
        {
            w(row,col) /= wmax;
            ainv(row,col) /= wmax;
        }
    }

    // Gauss-Jordan elimination with partial pivoting
    for(col=0;col<n;col++)
    {
        //find location of max in each column
        wmax=0.0;
        startrow = -1;
        for(row=col;row<n;row++)
        {
            if(std::abs(w(row,col))>wmax)
            {
                wmax = std::abs(w(row,col));
                
                startrow=row;
            }
        }

        for(int j=col;j<n;j++) dummy(j) = w(col,j);
        for(int j=col;j<n;j++) w(col,j) = w(startrow,j);
        for(int j=col;j<n;j++) w(startrow,j) = dummy(j);
        for(int j=0;j<n;j++) dummy(j) = ainv(col,j);
        for(int j=0;j<n;j++) ainv(col,j) = ainv(startrow,j);
        for(int j=0;j<n;j++) ainv(startrow,j) = dummy(j);

        wmax = w(col,col);

        //check for ill conditioned tensor
        if(std::abs(wmax) < wcond)
        {
            icond = 1;
            throw std::invalid_argument(
                    "p3a_mandel6x6::Inverse: illconditioned 4th order Mandel Tensor");
        }

        row = col;
        for(int j=col;j<n;j++) w[row,j] /= wmax;
        for(int k=0;k<n;k++) ainv[row,k] /= wmax;

        for(row=0;row<n;row++)
        {
            if(row == col) continue;
            fac = w[row*n+col];
            for(int j=col;j<n;j++) w[row,j] -= fac * w[col,j];
            for(int k=0;k<n;k++) ainv[row,k] -= fac * ainv[col,k];
        }//end row loop

    }//end col loop

    return mandel6x6<U>(ainv,true);
}

/** Create a 6x6 Mandel Tensor to Rotate a 6x6 Mandel Tensor
 *
 *	Convert 3x3 rotation operator to rotation operator for Mandel Vectors.
 *	This Returns a tensor in Mandel Notation!
 *
 *  @param a Rotation Operator Tensor
 *
 *  @return Rotation Operator for Mandel as MandelTensor
 */
template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
mandel6x6<T> MandelTransformation(
    matrix3x3<T> const& a)
{
    return mandel6x6<T>(
        a.xx()*a.xx(),
        a.xy()*a.xy(),
        a.xz()*a.xz(),
        a.xy()*a.xz()*r2,
        a.xx()*a.xz()*r2,
        a.xx()*a.xy()*r2,
        a.yx()*a.yx(),
        a.yy()*a.yy(),
        a.yz()*a.yz(),
        a.yy()*a.yz()*r2,
        a.yx()*a.yz()*r2,
        a.yx()*a.yy()*r2,
        a.zx()*a.zx(),
        a.zy()*a.zy(),
        a.zz()*a.zz(),
        a.zy()*a.zz()*r2,
        a.zx()*a.zz()*r2,
        a.zx()*a.zy()*r2,
        a.yx()*a.zx()*r2,
        a.yy()*a.zy()*r2,
        a.yz()*a.zz()*r2,
        a.yy()*a.zz()+a.yz()*a.zy(),
        a.yz()*a.zx()+a.yx()*a.zz(),
        a.yx()*a.zy()+a.yy()*a.zx(),
        a.xx()*a.zx()*r2,
        a.xy()*a.zy()*r2,
        a.xz()*a.zz()*r2,
        a.zy()*a.xz()+a.zz()*a.xy(),
        a.zz()*a.xx()+a.zx()*a.xz(),
        a.zx()*a.xy()+a.zy()*a.xx(),
        a.xx()*a.yx()*r2,
        a.xy()*a.yy()*r2,
        a.xz()*a.yz()*r2,
        a.xy()*a.yz()+a.xz()*a.yy(),
        a.xz()*a.yx()+a.xx()*a.yz(),
        a.xx()*a.yy()+a.xy()*a.yx(),
        false); 
}

/** transpose **/
template <class T>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
mandel6x6<T> transpose(
    mandel6x6<T> const &d)
{
    return mandel6x6<T>(
        d.x11(), d.x21(), d.x31(), d.x41(), d.x51(), d.x61(),
        d.x12(), d.x22(), d.x32(), d.x42(), d.x52(), d.x62(),
        d.x13(), d.x23(), d.x33(), d.x43(), d.x53(), d.x63(),
        d.x14(), d.x24(), d.x34(), d.x44(), d.x54(), d.x64(),
        d.x15(), d.x25(), d.x35(), d.x45(), d.x55(), d.x65(),
        d.x16(), d.x26(), d.x36(), d.x46(), d.x56(), d.x66(),
        false); //already transformed
}

/** Tensor multiply mandel6x6 (6x6) by mandel6x1 (6x1) **/
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(
    mandel6x6<T> const &C,
    mandel6x1<U> const &v)
{
    return mandel6x1<decltype(C.x11*v.x1)>(
        (C.x11()*v.x1() + C.x12()*v.x2() + C.x13()*v.x3() + C.x14()*v.x4() + C.x15()*v.x5() + C.x16()*v.x6()),
        (C.x21()*v.x1() + C.x22()*v.x2() + C.x23()*v.x3() + C.x24()*v.x4() + C.x25()*v.x5() + C.x26()*v.x6()),
        (C.x31()*v.x1() + C.x32()*v.x2() + C.x33()*v.x3() + C.x34()*v.x4() + C.x35()*v.x5() + C.x36()*v.x6()),
        (C.x41()*v.x1() + C.x42()*v.x2() + C.x43()*v.x3() + C.x44()*v.x4() + C.x45()*v.x5() + C.x46()*v.x6()),
        (C.x51()*v.x1() + C.x52()*v.x2() + C.x53()*v.x3() + C.x54()*v.x4() + C.x55()*v.x5() + C.x56()*v.x6()),
        (C.x61()*v.x1() + C.x62()*v.x2() + C.x63()*v.x3() + C.x64()*v.x4() + C.x65()*v.x5() + C.x66()*v.x6()),
        false); //already transformed
}

/** Tensor multiply Mandel6x6 (6x6) by symmetric3x3 (3x3) **/
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(
    mandel6x6<T> const &C, 
    symmetric3x3<U> const &s)
{
    v = mandel6x1<T>(s);
    return C*v;
}

/** Tensor multiply MandelVector (6x6) by diagonal3x3 (3x3 as 6x1) **/
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(
    mandel6x6<T> const &C, 
    diagonal3x3<U> const &t)
{
    return mandel6x1<decltype(t.xx() * C.x11())>(
          C.x11()*t.xx() + C.x12()*t.yy() + C.x13()*t.zz(),
          C.x21()*t.xx() + C.x22()*t.yy() + C.x23()*t.zz(),
          C.x31()*t.xx() + C.x32()*t.yy() + C.x33()*t.zz(),
          C.x41()*t.xx() + C.x42()*t.yy() + C.x43()*t.zz(),
          C.x51()*t.xx() + C.x52()*t.yy() + C.x53()*t.zz(),
          C.x61()*t.xx() + C.x62()*t.yy() + C.x63()*t.zz(),
          false);
}

//misc
inline int constexpr mandel6x6_component_count = 36;
