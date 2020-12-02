#pragma once

#include "p3a_macros.hpp"

namespace p3a {

P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
char* copy_c_string(char* dest, char const* src)
{
  char* const result = dest;
  while (*src != '\0') {
    *dest = *src;
    ++dest;
    ++src;
  }
  *dest = '\0';
  return result;
}

P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
char* append_c_string(char* dest, char const* src)
{
  char* const result = dest;
  while (*dest != '\0') {
    ++dest;
  }
  copy_c_string(dest, src);
  return result;
}

P3A_HOST P3A_DEVICE P3A_ALWAYS_INLINE inline constexpr
int compare_c_strings(char const* s1, char const* s2)
{
  while ((*s1 != '\0') && (*s1 == *s2)) {
    ++s1;
    ++s2;
  }
  return int(*s1) - int(*s2);
}

}
