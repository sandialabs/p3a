#include <Kokkos_Core.hpp>
#include "gtest/gtest.h"

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::ScopeGuard kokkos_library_state(argc, argv);
  return RUN_ALL_TESTS();
}
