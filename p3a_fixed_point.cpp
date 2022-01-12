#include "p3a_fixed_point.hpp"

extern "C" void p3a_mpi_int128_sum(
    void* a,
    void* b,
    int*,
    MPI_Datatype*)
{
  p3a::int128* a2 = static_cast<p3a::int128*>(a);
  p3a::int128* b2 = static_cast<p3a::int128*>(b);
  *b2 = *b2 + *a2;
}
