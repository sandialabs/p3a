Portably Performant Physical Algebra
====================================

This is a C++17 library that is meant to support High Performance Computing physics applications.
P3A is meant primarily to help structured grid applications perform well on Intel CPUs, NVIDIA GPUs, and AMD GPUs, but can be used for other applications and can be extended for other hardware.
P3A provides several "layers" of helpful tools:

2. A `p3a::dynamic_array` class that is very similar to `std::vector`, except for an `ExecutionPolicy` template argument that helps it work on GPUs.
3. A few algorithms modeled after C++17's parallel algorithms library such as `p3a::for_each`, `p3a::transform_reduce`, and `p3a::copy`. These include non-standard overloads for iterating over 3D grids.
4. A set of tensor algebra types such as `p3a::symmetric3x3`, `p3a::vector3`, and `p3a::matrix3x3`, with relevant mathematical operator overloads and load/store helpers.
5. A `p3a::quantity<T, Dimension>` scalar type that performs compile-time dimensional analysis on all physics code.
6. A set of SIMD scalar types `p3a::simd<T, Abi>` that directly call CPU vector intrinsics. Currently this is limited to one for Intel AVX-512 and one that is just a scalar.

The ideal design for an application using P3A is to implement a physics code as a series of loops which perform outer loop vectorization, while keeping the C++ source code readable by using tensor types with overloaded mathematical operators.

P3A depends on [Kokkos](https://github.com/kokkos/kokkos) for its parallel algorithms,
and the [MPICPP](https://cee-gitlab.sandia.gov/1443-public/code/mpicpp) for its MPI interface.

At Sandia, P3A is SCR 2619.0
