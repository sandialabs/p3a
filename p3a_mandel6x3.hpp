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
class mandel6x3
/** 
 * Represents a 3th order tensor as a 6x3 Mandel array
 */
/******************************************************************/
{
 T m_x11,m_x12,m_x13,
   m_x21,m_x22,m_x23,
   m_x31,m_x32,m_x33,
   m_x41,m_x42,m_x43,
   m_x51,m_x52,m_x53,
   m_x61,m_x62,m_x63;
 bool applyTransform;

 public:
  static constexpr T r2 = square_root_of_two_value<T>();

  static constexpr T r2i = T(1.0)/square_root_of_two_value<T>();

  static constexpr T two = T(2.0);

  /**** constructors, destructors, and assigns ****/
  P3A_ALWAYS_INLINE constexpr
  mandel6x3() = default;

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  mandel6x3(
      T const& X11, T const& X12, T const& X13,
      T const& X21, T const& X22, T const& X23,
      T const& X31, T const& X32, T const& X33,
      T const& X41, T const& X42, T const& X43,
      T const& X51, T const& X52, T const& X53,
      T const& X61, T const& X62, T const& X63):
      m_x11(X11),m_x12(X12),m_x13(X13),
      m_x21(X21),m_x22(X22),m_x23(X23),
      m_x31(X31),m_x32(X32),m_x33(X33),
      m_x41(X41),m_x42(X42),m_x43(X43),
      m_x51(X51),m_x52(X52),m_x53(X53),
      m_x61(X61),m_x62(X62),m_x63(X63),
       applyTransform(true)
  {
    this->MandelXform();
  }

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  mandel6x3(
      T const& X11, T const& X12, T const& X13,
      T const& X21, T const& X22, T const& X23,
      T const& X31, T const& X32, T const& X33,
      T const& X41, T const& X42, T const& X43,
      T const& X51, T const& X52, T const& X53,
      T const& X61, T const& X62, T const& X63, 
      bool const& Xform):
    m_x11(X11),m_x12(X12),m_x13(X13),
    m_x21(X21),m_x22(X22),m_x23(X23),
    m_x31(X31),m_x32(X32),m_x33(X33),
    m_x41(X41),m_x42(X42),m_x43(X43),
    m_x51(X51),m_x52(X52),m_x53(X53),
    m_x61(X61),m_x62(X62),m_x63(X63),
    applyTransform(Xform)
  {
    if (applyTransform)
        this->MandelXform();
  }

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  mandel6x3(
      static_matrix<T,6,3> const& X, bool const& Xform):
    m_x11(X(0,0)),m_x12(X(0,1)),m_x13(X(0,2)),
    m_x21(X(1,0)),m_x22(X(1,1)),m_x23(X(1,2)),
    m_x31(X(2,0)),m_x32(X(2,1)),m_x33(X(2,2)),
    m_x41(X(3,0)),m_x42(X(3,1)),m_x43(X(3,2)),
    m_x51(X(4,0)),m_x52(X(4,1)),m_x53(X(4,2)),
    m_x61(X(5,0)),m_x62(X(5,1)),m_x63(X(5,2)),
    applyTransform(Xform)
  {
    if (applyTransform)
        this->MandelXform();
  }

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  mandel6x3(
      static_matrix<T,6,3> const& X):
    m_x11(X(0,0)),m_x12(X(0,1)),m_x13(X(0,2)),
    m_x21(X(1,0)),m_x22(X(1,1)),m_x23(X(1,2)),
    m_x31(X(2,0)),m_x32(X(2,1)),m_x33(X(2,2)),
    m_x41(X(3,0)),m_x42(X(3,1)),m_x43(X(3,2)),
    m_x51(X(4,0)),m_x52(X(4,1)),m_x53(X(4,2)),
    m_x61(X(5,0)),m_x62(X(5,1)),m_x63(X(5,2)),
    applyTransform(true)
  {
    this->MandelXform();
  }

  //return by mandel index 1-6
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x11() const { return m_x11; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x12() const { return m_x12; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x13() const { return m_x13; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x21() const { return m_x21; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x22() const { return m_x22; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x23() const { return m_x23; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x31() const { return m_x31; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x32() const { return m_x32; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x33() const { return m_x33; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x41() const { return m_x41; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x42() const { return m_x42; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x43() const { return m_x43; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x51() const { return m_x51; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x52() const { return m_x52; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x53() const { return m_x53; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x61() const { return m_x61; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x62() const { return m_x62; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T const& x63() const { return m_x63; }

  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x11() { return m_x11; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x12() { return m_x12; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x13() { return m_x13; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x21() { return m_x21; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x22() { return m_x22; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x23() { return m_x23; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x31() { return m_x31; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x32() { return m_x32; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x33() { return m_x33; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x41() { return m_x41; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x42() { return m_x42; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x43() { return m_x43; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x51() { return m_x51; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x52() { return m_x52; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x53() { return m_x53; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x61() { return m_x61; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x62() { return m_x62; }
  [[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  T& x63() { return m_x63; }

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE static constexpr
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

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE static constexpr
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

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  void MandelXform()
  {
       m_x41*=r2i, m_x42*=r2i, m_x43*=r2i,
       m_x51*=r2i, m_x52*=r2i, m_x53*=r2i,
       m_x61*=r2i, m_x62*=r2i, m_x63*=r2i;
  }

  P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
  void invMandelXform()
  {
       m_x41/=r2i, m_x42/=r2i, m_x43/=r2i,
       m_x51/=r2i, m_x52/=r2i, m_x53/=r2i,
       m_x61/=r2i, m_x62/=r2i, m_x63/=r2i;
  }

};

/***************************************************************************** 
 *****************************************************************************/
//mandel6x3 binary operators with scalars
//multiplication by constant
template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<B>, mandel6x3<decltype(A() * B())>>::type
operator*(
        mandel6x3<A> const& a, 
        B const& c)
{
    return mandel6x3<decltype(A()*B())>(
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
typename std::enable_if<is_scalar<A>, mandel6x3<decltype(A() * B())>>::type 
operator*(
        A const& c, 
        mandel6x3<B> const& t)
{
    return t * c;
}

//division by constant
template <class A, class B>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
typename std::enable_if<is_scalar<B>, mandel6x3<decltype(A() * B())>>::type
operator/(
        mandel6x3<A> const& a, 
        B const& c)
{
    return mandel6x3<decltype(A()*B())>(
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
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
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
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
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

//mandel6x3 += addition
template <class T>
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
void operator+=(
    mandel6x3<T>& a, 
    mandel6x3<T> const& b)
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
P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
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

/** Tensor multiply mandel6x3 (6x3) by mandel3x6 (3x6) **/
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(
    mandel6x3<T> const &e,
    mandel3x6<U> const &d)
{
    return mandel6x6<decltype(e.x11()*d.x11())>(
        e.x11()*d.x11() + e.x12()*d.x21() + e.x13()*d.x31(),
        e.x11()*d.x12() + e.x12()*d.x22() + e.x13()*d.x32(),
        e.x11()*d.x13() + e.x12()*d.x23() + e.x13()*d.x33(),
        e.x11()*d.x14() + e.x12()*d.x24() + e.x13()*d.x34(),
        e.x11()*d.x15() + e.x12()*d.x25() + e.x13()*d.x35(),
        e.x11()*d.x16() + e.x12()*d.x26() + e.x13()*d.x36(),
        e.x21()*d.x11() + e.x22()*d.x21() + e.x23()*d.x31(),
        e.x21()*d.x12() + e.x22()*d.x22() + e.x23()*d.x32(),
        e.x21()*d.x13() + e.x22()*d.x23() + e.x23()*d.x33(),
        e.x21()*d.x14() + e.x22()*d.x24() + e.x23()*d.x34(),
        e.x21()*d.x15() + e.x22()*d.x25() + e.x23()*d.x35(),
        e.x21()*d.x16() + e.x22()*d.x26() + e.x23()*d.x36(),
        e.x31()*d.x11() + e.x32()*d.x21() + e.x33()*d.x31(),
        e.x31()*d.x12() + e.x32()*d.x22() + e.x33()*d.x32(),
        e.x31()*d.x13() + e.x32()*d.x23() + e.x33()*d.x33(),
        e.x31()*d.x14() + e.x32()*d.x24() + e.x33()*d.x34(),
        e.x31()*d.x15() + e.x32()*d.x25() + e.x33()*d.x35(),
        e.x31()*d.x16() + e.x32()*d.x26() + e.x33()*d.x36(),
        e.x41()*d.x11() + e.x42()*d.x21() + e.x43()*d.x31(),
        e.x41()*d.x12() + e.x42()*d.x22() + e.x43()*d.x32(),
        e.x41()*d.x13() + e.x42()*d.x23() + e.x43()*d.x33(),
        e.x41()*d.x14() + e.x42()*d.x24() + e.x43()*d.x34(),
        e.x41()*d.x15() + e.x42()*d.x25() + e.x43()*d.x35(),
        e.x41()*d.x16() + e.x42()*d.x26() + e.x43()*d.x36(),
        e.x51()*d.x11() + e.x52()*d.x21() + e.x53()*d.x31(),
        e.x51()*d.x12() + e.x52()*d.x22() + e.x53()*d.x32(),
        e.x51()*d.x13() + e.x52()*d.x23() + e.x53()*d.x33(),
        e.x51()*d.x14() + e.x52()*d.x24() + e.x53()*d.x34(),
        e.x51()*d.x15() + e.x52()*d.x25() + e.x53()*d.x35(),
        e.x51()*d.x16() + e.x52()*d.x26() + e.x53()*d.x36(),
        e.x61()*d.x11() + e.x62()*d.x21() + e.x63()*d.x31(),
        e.x61()*d.x12() + e.x62()*d.x22() + e.x63()*d.x32(),
        e.x61()*d.x13() + e.x62()*d.x23() + e.x63()*d.x33(),
        e.x61()*d.x14() + e.x62()*d.x24() + e.x63()*d.x34(),
        e.x61()*d.x15() + e.x62()*d.x25() + e.x63()*d.x35(),
        e.x61()*d.x16() + e.x62()*d.x26() + e.x63()*d.x36(),
        false);//already transformed
}

/** Tensor multiply mandel3x6 (3x6) by mandel6x3 (6x3) **/
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(
    mandel3x6<T> const &e,
    mandel6x3<U> const &d)
{
    return matrix3x3<decltype(e.x11()*d.x11())>(
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
    using result_type = decltype(e.x11()*f.x1());
    matrix3x3<result_type> d = mandel6x1_to_matrix3x3(f);
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
    using result_type = decltype(e.x11()*f.x11());
    mandel6x1<result_type> d(f);
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
    mandel6x3<T> const &e, 
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

/** Tensor Dot mandel6x6 (6x6) by mandel6x3 (6x3)*/
template <class T, class U>
[[nodiscard]] P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE constexpr
auto operator*(
    mandel6x6<T> const &e, 
    mandel6x3<U> const &d)
{
    return mandel6x3<decltype(e.x11()*d.x11())>(
            e.x11()*d.x11() + e.x12()*d.x21() + e.x13()*d.x31() + e.x14()*d.x41() + e.x15()*d.x51() + e.x16()*d.x61(), e.x11()*d.x12() + e.x12()*d.x22() + e.x13()*d.x32() + e.x14()*d.x42() + e.x15()*d.x52() + e.x16()*d.x62(), e.x11()*d.x13() + e.x12()*d.x23() + e.x13()*d.x33() + e.x14()*d.x43() + e.x15()*d.x53() + e.x16()*d.x63(), 
            e.x21()*d.x11() + e.x22()*d.x21() + e.x23()*d.x31() + e.x24()*d.x41() + e.x25()*d.x51() + e.x26()*d.x61(), e.x21()*d.x12() + e.x22()*d.x22() + e.x23()*d.x32() + e.x24()*d.x42() + e.x25()*d.x52() + e.x26()*d.x62(), e.x21()*d.x13() + e.x22()*d.x23() + e.x23()*d.x33() + e.x24()*d.x43() + e.x25()*d.x53() + e.x26()*d.x63(), 
            e.x31()*d.x11() + e.x32()*d.x21() + e.x33()*d.x31() + e.x34()*d.x41() + e.x35()*d.x51() + e.x36()*d.x61(), e.x31()*d.x12() + e.x32()*d.x22() + e.x33()*d.x32() + e.x34()*d.x42() + e.x35()*d.x52() + e.x36()*d.x62(), e.x31()*d.x13() + e.x32()*d.x23() + e.x33()*d.x33() + e.x34()*d.x43() + e.x35()*d.x53() + e.x36()*d.x63(), 
            e.x41()*d.x11() + e.x42()*d.x21() + e.x43()*d.x31() + e.x44()*d.x41() + e.x45()*d.x51() + e.x46()*d.x61(), e.x41()*d.x12() + e.x42()*d.x22() + e.x43()*d.x32() + e.x44()*d.x42() + e.x45()*d.x52() + e.x46()*d.x62(), e.x41()*d.x13() + e.x42()*d.x23() + e.x43()*d.x33() + e.x44()*d.x43() + e.x45()*d.x53() + e.x46()*d.x63(),
            e.x51()*d.x11() + e.x52()*d.x21() + e.x53()*d.x31() + e.x54()*d.x41() + e.x55()*d.x51() + e.x56()*d.x61(), e.x51()*d.x12() + e.x52()*d.x22() + e.x53()*d.x32() + e.x54()*d.x42() + e.x55()*d.x52() + e.x56()*d.x62(), e.x51()*d.x13() + e.x52()*d.x23() + e.x53()*d.x33() + e.x54()*d.x43() + e.x55()*d.x53() + e.x56()*d.x63(), 
            e.x61()*d.x11() + e.x62()*d.x21() + e.x63()*d.x31() + e.x64()*d.x41() + e.x65()*d.x51() + e.x66()*d.x61(), e.x61()*d.x12() + e.x62()*d.x22() + e.x63()*d.x32() + e.x64()*d.x42() + e.x65()*d.x52() + e.x66()*d.x62(), e.x61()*d.x13() + e.x62()*d.x23() + e.x63()*d.x33() + e.x64()*d.x43() + e.x65()*d.x53() + e.x66()*d.x63(),
            false); //already transformed
}


//misc
inline int constexpr mandel6x3_component_count = 18;

//output print
template <class U>
P3A_ALWAYS_INLINE constexpr 
std::ostream& operator<<(std::ostream& os, mandel6x3<U> const& a)
{

    os << std::cout.precision(4);
    os << std::scientific;
    std::cout << "\t  | " << a.x1()       << " " << a.x6()*a.r2i << " " << a.x5()*a.r2i << " |" <<std::endl;
    std::cout << "\t  | " << a.x6()*a.r2i << " " << a.x2()       << " " << a.x4()*a.r2i << " |" <<std::endl;
    std::cout << "\t  | " << a.x5()*a.r2i << " " << a.x4()*a.r2i << " " << a.x3()       << " |" <<std::endl;

    return os;

}

}
