#pragma once

namespace boost {
template <typename T>
std::vector<T> get_whole_vector(std::vector<T> const& v_local, MPI_Comm comm_) {
  boost::mpi::communicator comm{comm_, mpi::comm_attach};
  std::vector<T> v_global;
  std::vector<int> sizes;
  boost::mpi::all_gather(comm, static_cast<int>(v_local.size()), sizes);
  boost::mpi::all_gatherv(comm, v_local, v_global, sizes);
  return v_global;
}
}  // namespace boost

