#pragma once

#include "bfs/common.hpp"
#include "kamping/mpi_datatype.hpp"

namespace helper = kamping;

namespace bfs_mpi {
//> START BFS MPI
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
    graph::VertexBuffer data;
    std::vector<int> sCounts(size);
    for (size_t i = 0; i < size; ++i) {
      auto it = _data.find(i);
      if (it == _data.end()) {
        sCounts[i] = 0;
        continue;
      }
      auto &local_data = it->second;
      data.insert(data.end(), local_data.begin(), local_data.end());
      sCounts[i] = local_data.size();
    }
    _data.clear();
    std::vector<int> rCounts(size);
    MPI_Alltoall(sCounts.data(), 1, MPI_INT, rCounts.data(), 1, MPI_INT, _comm);
    std::vector<int> sDispls(size);
    std::vector<int> rDispls(size);
    std::exclusive_scan(sCounts.begin(), sCounts.end(), sDispls.begin(), 0);
    std::exclusive_scan(rCounts.begin(), rCounts.end(), rDispls.begin(), 0);
    const size_t num_recv_elems =
        static_cast<size_t>(rCounts.back() + rDispls.back());
    graph::VertexBuffer new_frontier(num_recv_elems);
    MPI_Alltoallv(data.data(), sCounts.data(), sDispls.data(),
                  helper::mpi_datatype<graph::VertexId>(), new_frontier.data(),
                  rCounts.data(), rDispls.data(),
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
//> END
}  // namespace bfs_mpi
