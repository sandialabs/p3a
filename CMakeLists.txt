cmake_minimum_required(VERSION 3.12.0...3.21.0)

project(p3a VERSION 1.0.0 LANGUAGES CXX)

enable_testing()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(p3a_ENABLE_AVX512 "Enable AVX-512" OFF)
set(p3a_ARCHITECTURES "" CACHE STRING "GPU architectures (either CUDA_ARCHITECTURES or HIP_ARCHITECTURES)")

find_package(MPI REQUIRED)
find_package(Kokkos REQUIRED COMPONENTS separable_compilation)

set(p3a_ENABLE_CUDA "${Kokkos_ENABLE_CUDA}")
set(p3a_ENABLE_HIP "${Kokkos_ENABLE_HIP}")

if (p3a_ENABLE_CUDA AND p3a_ENABLE_HIP)
  message(FATAL_ERROR "CUDA and HIP cannot be simultaneously enabled")
endif()

set(p3a_LANGUAGE "${Kokkos_COMPILE_LANGUAGE}")

enable_language(${p3a_LANGUAGE})

if (BUILD_TESTING)
  find_package(GTest REQUIRED)
endif()

set(P3A_HEADERS
  p3a_macros.hpp
  p3a_static_array.hpp
  p3a_constants.hpp
  p3a_functions.hpp
  p3a_vector2.hpp
  p3a_vector3.hpp
  p3a_static_vector.hpp
  p3a_box3.hpp
  p3a_grid3.hpp
  p3a_scaled_identity3x3.hpp
  p3a_diagonal3x3.hpp
  p3a_symmetric3x3.hpp
  p3a_matrix2x2.hpp
  p3a_matrix3x3.hpp
  p3a_execution.hpp
  p3a_dynamic_array.hpp
  p3a_for_each.hpp
  p3a_reduce.hpp
  p3a_quantity.hpp
  p3a_scalar.hpp
  p3a_identity3x3.hpp
  p3a_memory.hpp
  p3a_allocator.hpp
  p3a_dimension.hpp
  p3a_unit.hpp
  p3a_cstring.hpp
  p3a_mpi.hpp
  p3a_avx512.hpp
  p3a_simd.hpp
  p3a_simd_common.hpp
  p3a_simd_scalar.hpp
  p3a_int128.hpp
  p3a_counting_iterator.hpp
  p3a_dynamic_matrix.hpp
  p3a_skew3x3.hpp
  p3a_static_matrix.hpp
  p3a_eigen.hpp
  p3a_svd.hpp
  p3a_lie.hpp
  p3a_opts.hpp
  p3a_mandel6x1.hpp
  p3a_mandel6x6.hpp
  p3a_mandel3x6.hpp
  p3a_mandel6x3.hpp
  p3a_tuple.hpp
  p3a_axis_angle.hpp
  )

set(P3A_SOURCES
  p3a_execution.cpp
  p3a_mpi.cpp
  p3a_opts.cpp
  )

set_source_files_properties(
  ${P3A_SOURCES} PROPERTIES LANGUAGE ${p3a_LANGUAGE})
add_library(p3a ${P3A_SOURCES})
target_compile_features(p3a PUBLIC cxx_std_17)
set_target_properties(p3a PROPERTIES
  PUBLIC_HEADER "${P3A_HEADERS}")
target_include_directories(p3a
  PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
set_target_properties(p3a PROPERTIES ${p3a_LANGUAGE}_ARCHITECTURES "${p3a_ARCHITECTURES}")

if (p3a_ENABLE_AVX512)
  target_compile_options(p3a PUBLIC -mavx512f -mavx512vl)
endif()

target_link_libraries(p3a PUBLIC MPI::MPI_CXX)
target_link_libraries(p3a PUBLIC Kokkos::kokkoscore)

install(
  TARGETS p3a
  EXPORT p3a-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if (BUILD_TESTING)
  set_source_files_properties(
    p3a_unit_tests.cpp PROPERTIES LANGUAGE ${p3a_LANGUAGE})
  add_executable(p3a-unit-tests p3a_unit_tests.cpp)
  target_link_libraries(p3a-unit-tests PRIVATE p3a)
  target_link_libraries(p3a-unit-tests PRIVATE GTest::gtest_main)
  add_test(NAME unit-tests COMMAND p3a-unit-tests)
endif()

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/p3a-config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/p3a
)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/p3a-config-version.cmake"
  COMPATIBILITY AnyNewerVersion)

install(FILES
  "${PROJECT_BINARY_DIR}/p3a-config.cmake"
  "${PROJECT_BINARY_DIR}/p3a-config-version.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/p3a)

install(
  EXPORT p3a-targets
  NAMESPACE p3a::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/p3a)
