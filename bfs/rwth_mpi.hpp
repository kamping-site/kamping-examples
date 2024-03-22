#pragma once

#include <mpi/all.hpp>

#include "common.hpp"

namespace bfs_rwth_mpi {
class BFSFrontier final : public graph::BFSFrontier {
 public:
  BFSFrontier(MPI_Comm comm) : _comm{comm} {}
  std::pair<graph::VertexBuffer, bool> exchange() override {
    if (is_empty()) {
      return std::make_pair(graph::VertexBuffer{}, true);
    }
    graph::VertexBuffer send_buffer;
    std::vector<int> send_counts(static_cast<size_t>(_comm.size()));
    for (size_t rank = 0; rank < static_cast<size_t>(_comm.size()); rank++) {
      auto it = _data.find(static_cast<int>(rank));
      if (it == _data.end()) {
        send_counts[rank] = 0;
        continue;
      }
      auto &local_data = it->second;
      send_buffer.insert(send_buffer.end(), local_data.begin(),
                         local_data.end());
      send_counts[rank] = static_cast<int>(local_data.size());
    }
    _data.clear();
    graph::VertexBuffer new_frontier;
    _comm.all_to_all_varying(send_buffer, send_counts, new_frontier, true);
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
}  // namespace rwth_mpi
