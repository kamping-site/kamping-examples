#pragma once

#include "bfs/common.hpp"
#include <mpi/all.hpp>


namespace bfs_rwth_mpi {
class BFSFrontier final : public graph::BFSFrontier {
 public:
  BFSFrontier(MPI_Comm comm) : _comm{comm} {}
  std::pair<graph::VertexBuffer, bool> exchange() override {
    if (is_empty()) {
      return std::make_pair(graph::VertexBuffer{}, true);
    }
    graph::VertexBuffer data;
    std::vector<int> sCounts(_comm.size());
    for (size_t rank = 0; rank < _comm.size(); rank++) {
      auto it = _data.find(rank);
      if (it == _data.end()) {
        sCounts[rank] = 0;
        continue;
      }
      auto &local_data = it->second;
      data.insert(data.end(), local_data.begin(), local_data.end());
      sCounts[rank] = local_data.size();
    }
    _data.clear();
    graph::VertexBuffer new_frontier;
    _comm.all_to_all_varying(data, sCounts, new_frontier, true);
    return std::make_pair(std::move(new_frontier), false);
  }
  bool is_empty() const {
    bool is_empty = _data.empty();
    _comm.all_reduce(is_empty, mpi::ops::logical_and);
    return is_empty;
  }

 private:
  mpi::communicator _comm;
};
}  // namespace bfs_rwth_mpi
