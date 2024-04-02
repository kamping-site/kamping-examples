#pragma once

namespace kamping {
template <typename T>
std::vector<T> get_whole_vector(std::vector<T> const& v_local, MPI_Comm comm_) {
  kamping::Communicator comm(comm_);
  //> START VECTOR_ALLGATHER KAMPING
  return comm.allgatherv(kamping::send_buf(v_local));
  //> END
}
}  // namespace kamping
