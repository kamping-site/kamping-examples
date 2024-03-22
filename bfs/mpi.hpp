#pragma once

#include "common.hpp"
#include "kamping/mpi_datatype.hpp"

namespace helper = kamping;

namespace bfs_mpi {
class BFSFrontier final : public graph::BFSFrontier {
 public:
  BFSFrontier(MPI_Comm comm) : _comm{comm} {}
  std::pair<graph::VertexBuffer, bool> exchange() override {
    if (is_empty()) {
      return std::make_pair(graph::VertexBuffer{}, true);
    }
    int rank, size;
    MPI_Comm_rank(_comm, &rank);
    MPI_Comm_size(_comm, &size);
    graph::VertexBuffer send_buffer;
    std::vector<int> send_counts(size);
    for (size_t i = 0; i < size; ++i) {
      auto it = _data.find(i);
      if (it == _data.end()) {
        send_counts[i] = 0;
        continue;
      }
      auto &local_data = it->second;
      send_buffer.insert(send_buffer.end(), local_data.begin(),
                         local_data.end());
      send_counts[i] = local_data.size();
    }
    _data.clear();
    std::vector<int> recv_counts(size);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
                 _comm);
    std::vector<int> send_displs(size);
    std::vector<int> recv_displs(size);
    std::exclusive_scan(send_counts.begin(), send_counts.end(),
                        send_displs.begin(), 0);
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(),
                        recv_displs.begin(), 0);
    const size_t num_recv_elems =
        static_cast<size_t>(recv_counts.back() + recv_displs.back());
    graph::VertexBuffer new_frontier(num_recv_elems);
    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(),
                  helper::mpi_datatype<graph::VertexId>(), new_frontier.data(),
                  recv_counts.data(), recv_displs.data(),
                  helper::mpi_datatype<graph::VertexId>(), _comm);
    return std::make_pair(std::move(new_frontier), false);
  }

  bool is_empty() const {
    bool result = _data.empty();
    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_CXX_BOOL, MPI_LAND, _comm);
    return result;
  }

 private:
  MPI_Comm _comm;
};
}  // namespace bfs_mpi
