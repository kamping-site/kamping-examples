#pragma once

#include "common.hpp"
#include "mpi_neighborhood.hpp"

namespace bfs {

template <typename Frontier>
Frontier construct_frontier(const graph::Graph &g, MPI_Comm comm) {
  if constexpr (std::is_same_v<Frontier, bfs_mpi_neighborhood::BFSFrontier>) {
    return Frontier{comm, g.get_comm_partners()};
  } else {
    return Frontier{comm};
  }
}

template <typename Frontier>
std::vector<size_t> bfs(const graph::Graph &g, graph::VertexId root,
                        MPI_Comm comm) {
  using namespace graph;
  Frontier distributed_frontier = construct_frontier<Frontier>(g, comm);

  graph::VertexBuffer local_frontier;
  if (g.is_local(root)) {
    local_frontier.push_back(root);
  }
  std::vector<size_t> bfs_levels(g.vertex_end() - g.vertex_begin(),
                                 unreachable_vertex);
  size_t bfs_level = 0;
  bool has_finished = false;
  do {
    kamping::measurements::timer().synchronize_and_start(
        "local_frontier_processing");
    for (auto v : local_frontier) {
      if (bfs_levels[v - g.vertex_begin()] != unreachable_vertex) {
        continue;
      }
      bfs_levels[v - g.vertex_begin()] = bfs_level;
      for (const auto &[u, rank] : g.neighbors(v)) {
        distributed_frontier.add_vertex(u, rank);
      }
    }
    kamping::measurements::timer().stop_and_append();
    kamping::measurements::timer().synchronize_and_start("alltoall");
    std::tie(local_frontier, has_finished) = distributed_frontier.exchange();
    kamping::measurements::timer().stop_and_append();
    ++bfs_level;
  } while (!has_finished);
  return bfs_levels;
}

}  // namespace bfs
