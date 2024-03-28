#pragma once

namespace rwth {
template <typename T>
std::vector<T> get_whole_vector(std::vector<T> const& v_local, MPI_Comm comm_) {
  mpi::communicator comm(comm_);
  // std::vector<T> v_global = v_local;
  // // this requires manual patching of the code due to ambiguous overloads and
  // // does not properly resize the buffer
  // comm.all_gather_varying(v_global);
  // return v_global;
  std::vector<int> rc(static_cast<size_t>(comm.size()));
  comm.all_gather(static_cast<int>(v_local.size()), rc);
  std::vector<T> v_global;
  comm.all_gather_varying(v_local, v_global, rc, true);
  return v_global;
}
}  // namespace rwth
