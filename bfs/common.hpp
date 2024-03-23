#pragma once

#include <kagen/kagen.h>
#include <spdlog/fmt/ranges.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/reduce.hpp>
#include <kamping/communicator.hpp>
#include <kamping/environment.hpp>
#include <kamping/measurements/printer.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/mpi_datatype.hpp>
#include <kamping/plugin/alltoall_grid.hpp>
#include <kamping/plugin/alltoall_sparse.hpp>
#include <memory>
#include <random>
#include <ranges>
#include <tuple>

#include "../mpi_spd_formatters.hpp"

namespace graph {

using VertexId = kagen::SInt;
constexpr inline size_t unreachable_vertex = std::numeric_limits<size_t>::max();
using VertexBuffer = std::vector<VertexId>;

// Distributed graph data structure.
// Each rank is responsible ("home") for a subset of the overall vertices and
// their incident edges.
struct Graph {
  std::vector<VertexId> xadj;
  std::vector<VertexId> adjncy;
  std::vector<VertexId> vertex_distribution;
  kamping::Communicator<> const &comm;

  auto first_vertex() const { return vertex_distribution[comm.rank()]; }

  auto last_vertex() const { return vertex_distribution[comm.rank() + 1]; }

  bool is_local(VertexId v) const {
    return v >= first_vertex() && v < last_vertex();
  }

  int home_rank(VertexId v) const {
    auto rank = std::distance(vertex_distribution.begin(),
                              std::upper_bound(vertex_distribution.begin(),
                                               vertex_distribution.end(), v)) -
                1;
    return static_cast<int>(rank);
  }

  auto vertices() const {
    return std::ranges::views::iota(first_vertex(), last_vertex());
  }

  auto global_num_vertices() const { return vertex_distribution.back(); }

  auto neighbors(VertexId v) const {
    auto begin = xadj[v - first_vertex()];
    auto end = xadj[v - first_vertex() + 1];
    std::span span{adjncy};
    span = span.subspan(begin, end - begin);
    return span;
  }

  std::vector<int> get_comm_partners() const {
    std::unordered_set<int> comm_partners_set;
    std::vector<int> comm_partners;
    for (auto v : vertices()) {
      for (auto n : neighbors(v)) {
        comm_partners_set.insert(home_rank(n));
      }
    }
    for (const auto &v : comm_partners_set) {
      comm_partners.emplace_back(v);
    }
    std::sort(comm_partners.begin(), comm_partners.end());
    return comm_partners;
  }
};

inline auto generate_distributed_graph(const std::string &kagen_option_string) {
  kagen::KaGen kagen(MPI_COMM_WORLD);
  kagen.UseCSRRepresentation();
  auto graph = kagen.GenerateFromOptionString(kagen_option_string);
  std::vector xadj = graph.TakeXadj<graph::VertexId>();
  std::vector adjncy = graph.TakeAdjncy<graph::VertexId>();
  auto dist = kagen::BuildVertexDistribution<graph::VertexId>(
      graph, kamping::mpi_type_traits<graph::VertexId>::data_type(),
      MPI_COMM_WORLD);
  return Graph{std::move(xadj), std::move(adjncy), std::move(dist),
               kamping::comm_world()};
}

inline VertexId generate_start_vertex(const Graph &g, size_t seed = 0) {
  std::default_random_engine gen(seed);
  std::uniform_int_distribution<graph::VertexId> vertex_dist(
      0, g.global_num_vertices());
  vertex_dist(gen);
  vertex_dist(gen);
  return vertex_dist(gen);
}

/// @brief Represent the frontier in a distributed breadth-first search (BFS).
class BFSFrontier {
 public:
  void add_vertex(VertexId v, int rank) { _data[rank].push_back(v); }
  virtual std::pair<VertexBuffer, bool> exchange() = 0;
  virtual ~BFSFrontier() noexcept(false){};

 protected:
  std::unordered_map<int, std::vector<VertexId>>
      _data;  ///< map vertices of the frontier to their home rank
};

template <typename Frontier>
std::vector<size_t> bfs(const graph::Graph &g, graph::VertexId root,
                        MPI_Comm comm) {
  using namespace graph;
  Frontier distributed_frontier{comm};

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

template <typename Frontier>
void graph_ping_pong(const graph::Graph &g, MPI_Comm comm) {
  using namespace graph;

  Frontier distributed_frontier{comm};
  [[maybe_unused]] volatile bool b = false;
  for (size_t i = 0; i < 10; ++i) {
    for (const auto &v : g.vertices()) {
      for (const auto &u : g.neighbors(v)) {
        int rank = g.home_rank(u);
        distributed_frontier.add_vertex(u, rank);
      }
      auto result = distributed_frontier.exchange();
      b = result.second;
    }
  }
}

}  // namespace graph
