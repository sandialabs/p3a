cmake_minimum_required(VERSION 3.12.0...3.21.0)

project(p3a VERSION 1.0.0 LANGUAGES CXX)

if(WIN32 AND NOT CYGWIN)
  set(INSTALL_CMAKE_DIR CMake)
else()
  set(INSTALL_CMAKE_DIR lib/cmake/${PROJECT_NAME})
endif()

option(p3a_ENABLE_CUDA "Enable CUDA" OFF)
option(p3a_ENABLE_HIP "Enable HIP" OFF)

if (p3a_ENABLE_CUDA)
  enable_language(CUDA)
endif()

if (p3a_ENABLE_HIP)
  find_package(hip REQUIRED)
endif()

find_package(MPI REQUIRED)

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
  )

set(P3A_SOURCES
  p3a_execution.cpp
  p3a_mpi.cpp
  p3a_opts.cpp
  )

if (p3a_ENABLE_CUDA AND P3A_SOURCES)
  set_source_files_properties(
    ${P3A_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()

add_library(p3a ${P3A_SOURCES})
target_compile_features(p3a PUBLIC cxx_std_17)
set_target_properties(p3a PROPERTIES
  PUBLIC_HEADER "${P3A_HEADERS}")
target_include_directories(p3a
  INTERFACE $<INSTALL_INTERFACE:include>)

if (p3a_ENABLE_HIP)
  target_link_libraries(p3a PUBLIC hip::device)
endif()

target_link_libraries(p3a PUBLIC MPI::MPI_CXX)

install(TARGETS p3a EXPORT p3a-targets)

configure_file(p3a-config.cmake.in
  "${PROJECT_BINARY_DIR}/p3a-config.cmake" @ONLY)
configure_file(p3a-config-version.cmake.in
  "${PROJECT_BINARY_DIR}/p3a-config-version.cmake" @ONLY)

install(FILES
  "${PROJECT_BINARY_DIR}/p3a-config.cmake"
  "${PROJECT_BINARY_DIR}/p3a-config-version.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}")

install(EXPORT p3a-targets
    DESTINATION "${INSTALL_CMAKE_DIR}")
