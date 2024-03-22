#pragma once

#include "common.hpp"
#include "kamping/collectives/allreduce.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/plugin/alltoall_sparse.hpp"
#include "kamping/utils/flatten.hpp"

namespace kamping_sparse {
using namespace kamping;
class BFSFrontier final : public graph::BFSFrontier {
 public:
  BFSFrontier(MPI_Comm comm) : _comm{comm} {}
  std::pair<graph::VertexBuffer, bool> exchange() override {
    using namespace plugin::sparse_alltoall;
    if (is_empty()) {
      return std::make_pair(graph::VertexBuffer{}, true);
    }
    graph::VertexBuffer new_frontier;
    _comm.alltoallv_sparse(
        sparse_send_buf(_data), on_message([&](auto &probed_message) {
          auto old_size = static_cast<graph::VertexBuffer::difference_type>(
              new_frontier.size());
          new_frontier.resize(new_frontier.size() +
                              probed_message.recv_count());
          Span message{new_frontier.begin() + old_size, new_frontier.end()};
          probed_message.recv(kamping::recv_buf(message));
        }));
    return std::make_pair(std::move(new_frontier), false);
  }

  bool is_empty() const {
    return _comm.allreduce_single(send_buf(_data.empty()),
                                  op(ops::logical_and<bool>{}));
  }

 private:
  kamping::Communicator<std::vector, plugin::SparseAlltoall> _comm;
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
}  // namespace kamping_sparse
