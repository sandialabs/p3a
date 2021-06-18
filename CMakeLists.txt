cmake_minimum_required(VERSION 3.12.0...3.18.0)

project(P3A VERSION 0.1.0 LANGUAGES CXX)

if(WIN32 AND NOT CYGWIN)
  set(INSTALL_CMAKE_DIR CMake)
else()
  set(INSTALL_CMAKE_DIR lib/cmake/${PROJECT_NAME})
endif()

option(P3A_ENABLE_CUDA "Enable CUDA" OFF)

if (P3A_ENABLE_CUDA)
  enable_language(CUDA)
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
  p3a_symmetric3x3.hpp
  p3a_matrix2x2.hpp
  p3a_matrix3x3.hpp
  p3a_execution.hpp
  p3a_dynamic_array.hpp
  p3a_for_each.hpp
  p3a_reduce.hpp
  p3a_quantity.hpp
  p3a_polyhedron.hpp
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
  p3a_plane.hpp
  p3a_opts.hpp
  p3a_cg.hpp
  )

set(P3A_SOURCES
  p3a_execution.cpp
  p3a_mpi.cpp
  p3a_opts.cpp
  )

if (P3A_ENABLE_CUDA AND P3A_SOURCES)
  set_source_files_properties(
    ${P3A_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()

add_library(p3a ${P3A_SOURCES})
target_compile_features(p3a PUBLIC cxx_std_17)
set_target_properties(p3a PROPERTIES
  PUBLIC_HEADER "${P3A_HEADERS}")
target_include_directories(p3a
  INTERFACE $<INSTALL_INTERFACE:include>)

target_link_libraries(p3a PUBLIC MPI::MPI_CXX)

install(TARGETS p3a EXPORT P3ATargets)

configure_file(P3AConfig.cmake.in
  "${PROJECT_BINARY_DIR}/P3AConfig.cmake" @ONLY)
configure_file(P3AConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}/P3AConfigVersion.cmake" @ONLY)

install(FILES
  "${PROJECT_BINARY_DIR}/P3AConfig.cmake"
  "${PROJECT_BINARY_DIR}/P3AConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

install(EXPORT P3ATargets
    DESTINATION "${INSTALL_CMAKE_DIR}")
