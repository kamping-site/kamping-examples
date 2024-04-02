#pragma once

#include <mpi.h>

#include <kamping/mpi_datatype.hpp>
#include <vector>

namespace mpi {
template <typename T>
std::vector<T> get_whole_vector(std::vector<T> const& v_local, MPI_Comm comm) {
  //> START VECTOR_ALLGATHER MPI
  int size;
  int rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  std::vector<int> rc(size), rd(size);
  rc[rank] = v_local.size();
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rc.data(), 1, MPI_INT,
                comm);
  std::exclusive_scan(rc.begin(), rc.end(), rd.begin(), 0);
  std::vector<T> v_global(rd.back() + rc.back());
  MPI_Allgatherv(v_local.data(), v_local.size(), kamping::mpi_datatype<T>(),
                 v_global.data(), rc.data(), rd.data(),
                 kamping::mpi_datatype<T>(), comm);
  return v_global;
  //> END
}
}  // namespace mpi
