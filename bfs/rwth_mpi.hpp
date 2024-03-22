#pragma once

#include <mpi/all.hpp>

#include "common.hpp"

namespace rwth_mpi {
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

inline std::vector<size_t> bfs(const graph::Graph &g, graph::VertexId root,
                               MPI_Comm comm) {
  using namespace graph;
  kamping::measurements::timer().synchronize_and_start("bfs");
  BFSFrontier distributed_frontier{comm};

  graph::VertexBuffer local_frontier;
  if (g.is_local(root)) {
    local_frontier.push_back(root);
  }
  std::vector<size_t> bfs_levels(g.last_vertex() - g.first_vertex(),
                                 unreachable_vertex);
  size_t bfs_level = 0;
  bool has_finished = false;
  do {
    for (auto v : local_frontier) {
      if (bfs_levels[v - g.first_vertex()] != unreachable_vertex) {
        continue;
      }
      bfs_levels[v - g.first_vertex()] = bfs_level;
      for (auto u : g.neighbors(v)) {
        int rank = g.home_rank(u);
        distributed_frontier.add_vertex(u, rank);
      }
    }
    std::tie(local_frontier, has_finished) = distributed_frontier.exchange();
    ++bfs_level;
  } while (!has_finished);
  return bfs_levels;
}
}  // namespace rwth_mpi
