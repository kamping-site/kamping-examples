#pragma once

namespace kamping {
template <typename T>
std::vector<T> get_whole_vector(std::vector<T> const& v_local, MPI_Comm comm_) {
  using namespace kamping;
  kamping::Communicator comm(comm_);
  return comm.allgatherv(send_buf(v_local));
}
}  // namespace kamping
