#pragma once

namespace mpl {
//> START VECTOR_ALLGATHER MPL
template <typename T>
std::vector<T> get_whole_vector(std::vector<T> const& v_local, MPI_Comm) {
  auto const& comm = mpl::environment::comm_world();
  //> START VECTOR_ALLGATHER MPL
  std::vector<int> rc(comm.size());
  comm.allgather(static_cast<int>(v_local.size()), rc.data());
  mpl::layouts<T> recvls;
  int recv_displ = 0;
  for (int rank = 0; rank < comm.size(); ++rank) {
    recvls.push_back(mpl::indexed_layout<T>({{rc[rank], recv_displ}}));
    recv_displ += rc[rank];
  }
  std::vector<T> v_global(recv_displ);
  mpl::contiguous_layout<T> sendl(v_local.size());
  comm.allgatherv(v_local.data(), sendl, v_global.data(), recvls);
  return v_global;
  //> END
}
}  // namespace mpl
