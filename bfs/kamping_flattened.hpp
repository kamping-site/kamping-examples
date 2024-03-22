#pragma once

#include "common.hpp"
#include "kamping/collectives/allreduce.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/utils/flatten.hpp"

namespace kamping_flattened {
using namespace kamping;
class BFSFrontier final : public graph::BFSFrontier {
 public:
  BFSFrontier(MPI_Comm comm) : _comm{comm} {}
  std::pair<graph::VertexBuffer, bool> exchange() override {
    if (is_empty()) {
      return std::make_pair(graph::VertexBuffer{}, true);
    }
    auto new_frontier =
        with_flattened(_data, _comm.size()).call([&](auto... flattened) {
          _data.clear();
          return _comm.alltoallv(std::move(flattened)...);
        });
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
}  // namespace kamping_flattened
