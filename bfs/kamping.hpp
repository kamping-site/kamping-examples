#pragma once

#include "common.hpp"
#include "kamping/collectives/allreduce.hpp"
#include "kamping/collectives/alltoall.hpp"

namespace kamping {
class BFSFrontier final : public graph::BFSFrontier {
 public:
  BFSFrontier(MPI_Comm comm) : _comm{comm} {}
  std::pair<graph::VertexBuffer, bool> exchange() override {
    if (is_empty()) {
      return std::make_pair(graph::VertexBuffer{}, true);
    }
    graph::VertexBuffer send_buffer;
    std::vector<int> send_counts(_comm.size());
    for (size_t rank = 0; rank < _comm.size(); rank++) {
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
    auto new_frontier = _comm.alltoallv(send_buf(send_buffer),
                                        kamping::send_counts(send_counts));
    return std::make_pair(std::move(new_frontier), false);
  }

  bool is_empty() const {
    return _comm.allreduce_single(send_buf(_data.empty()),
                                  op(ops::logical_and<bool>{}));
  }

 private:
  kamping::Communicator<> _comm;
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
}  // namespace kamping
