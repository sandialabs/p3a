cmake_minimum_required(VERSION 3.12.0...3.18.0)

project(P3A VERSION 0.1.0 LANGUAGES CXX)

option(P3A_ENABLE_CUDA "Enable CUDA" OFF)

if(WIN32 AND NOT CYGWIN)
  set(INSTALL_CMAKE_DIR CMake)
  set(INSTALL_BIN_DIR ".")
  set(INSTALL_LIB_DIR ".")
else()
  set(INSTALL_CMAKE_DIR lib/cmake/${PROJECT_NAME})
  set(INSTALL_BIN_DIR "bin")
  set(INSTALL_LIB_DIR "lib")
endif()

if (P3A_ENABLE_CUDA)
  enable_language(CUDA)
endif()

set(P3A_HEADERS
  p3a.hpp
  p3a_macros.hpp
  p3a_array.hpp
  p3a_constants.hpp
  p3a_functions.hpp
  p3a_vector3.hpp
  p3a_box3.hpp
  p3a_grid3.hpp
  p3a_scaled_identity3x3.hpp
  p3a_symmetric3x3.hpp
  p3a_matrix3x3.hpp
  p3a_execution.hpp
  p3a_vector.hpp
  p3a_for_each.hpp
  p3a_reduce.hpp
  p3a_quantity.hpp
  p3a_polyhedron.hpp
  p3a_scalar.hpp
  p3a_identity3x3.hpp
  p3a_memory.hpp
  p3a_dimension.hpp
  )

add_library(p3a INTERFACE)
target_compile_features(p3a INTERFACE cxx_std_17)
set_target_properties(p3a PROPERTIES
  PUBLIC_HEADER "${P3A_HEADERS}")
target_include_directories(p3a
  INTERFACE $<INSTALL_INTERFACE:include>)

install(TARGETS p3a
  EXPORT P3ATargets
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}"
  RUNTIME DESTINATION "${INSTALL_LIB_DIR}"
  PUBLIC_HEADER DESTINATION "include"
  )

configure_file(P3AConfig.cmake.in
  "${PROJECT_BINARY_DIR}/P3AConfig.cmake" @ONLY)
configure_file(P3AConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}/P3AConfigVersion.cmake" @ONLY)

# Install the P3AConfig.cmake and P3AConfigVersion.cmake
install(FILES
  "${PROJECT_BINARY_DIR}/P3AConfig.cmake"
  "${PROJECT_BINARY_DIR}/P3AConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

install(EXPORT P3ATargets
    DESTINATION "${INSTALL_CMAKE_DIR}")
