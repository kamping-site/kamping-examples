#pragma once

#include <boost/mpi.hpp>

#include "bfs/common.hpp"

namespace bfs_boost {
class BFSFrontier final : public graph::BFSFrontier {
 public:
  BFSFrontier(MPI_Comm comm) : _comm{comm, boost::mpi::comm_attach} {}
  std::pair<graph::VertexBuffer, bool> exchange() override {
    if (is_empty()) {
      return std::make_pair(graph::VertexBuffer{}, true);
    }
    graph::VertexBuffer send_buffer;
    std::vector<int> send_counts(_comm.size());
    for (size_t rank = 0; rank < _comm.size(); rank++) {
      auto it = _data.find(rank);
      if (it == _data.end()) {
        send_counts[rank] = 0;
        continue;
      }
      auto &local_data = it->second;
      send_buffer.insert(send_buffer.end(), local_data.begin(),
                         local_data.end());
      send_counts[rank] = local_data.size();
    }
    _data.clear();
    std::vector<int> recv_counts(_comm.size());
    boost::mpi::all_to_all(_comm, send_counts, recv_counts);
    std::vector<int> send_displs(_comm.size());
    std::vector<int> recv_displs(_comm.size());
    std::exclusive_scan(send_counts.begin(), send_counts.end(),
                        send_displs.begin(), 0);
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(),
                        recv_displs.begin(), 0);
    graph::VertexBuffer new_frontier(recv_counts.back() + recv_displs.back());
    // Boost.MPI does not support alltoallv
    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(),
                  boost::mpi::get_mpi_datatype<graph::VertexId>(),
                  new_frontier.data(), recv_counts.data(), recv_displs.data(),
                  boost::mpi::get_mpi_datatype<graph::VertexId>(), _comm);
    return std::make_pair(std::move(new_frontier), false);
  }
  bool is_empty() const {
    return boost::mpi::all_reduce(_comm, _data.empty(), std::logical_and<>{});
  }

 private:
  boost::mpi::communicator _comm;
};
}  // namespace bfs_boost
