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
struct Edge {
  VertexId u;
  int rank;
};

// Distributed graph data structure.
// Each rank is responsible ("home") for a subset of the overall vertices and
// their incident edges.
class Graph {
 public:
  Graph(std::vector<VertexId> &&xadj, std::vector<Edge> &&adjncy,
        std::vector<VertexId> &&vertex_distribution,
        kamping::Communicator<> const &comm)
      : _rank{comm.rank()},
        _xadj{std::move(xadj)},
        _adjncy{std::move(adjncy)},
        _vertex_distribution{std::move(vertex_distribution)} {}
  Graph(std::vector<VertexId> &&xadj, std::vector<VertexId> &&adjncy,
        std::vector<VertexId> &&vertex_distribution,
        kamping::Communicator<> const &comm)
      : _rank{comm.rank()},
        _xadj{std::move(xadj)},
        _vertex_distribution{std::move(vertex_distribution)} {
    _home_rank.resize(_adjncy.size());
    auto compute_home_rank = [&](VertexId v) {
      auto rank =
          std::distance(_vertex_distribution.begin(),
                        std::upper_bound(_vertex_distribution.begin(),
                                         _vertex_distribution.end(), v)) -
          1;
      return static_cast<int>(rank);
    };
    // compute home rank for each edge endpoint u
    _adjncy.resize(adjncy.size());
    for (size_t i = 0; i < adjncy.size(); ++i) {
      auto u = adjncy[i];
      auto rank = compute_home_rank(u);
      _adjncy[i] = {u, rank};
    }
    std::vector<VertexId>{}.swap(adjncy);  // dump content of adjncy
  }

  auto vertex_begin() const { return _vertex_distribution[_rank]; }

  auto vertex_end() const { return _vertex_distribution[_rank + 1]; }

  bool is_local(VertexId v) const {
    return v >= vertex_begin() && v < vertex_end();
  }

  auto vertices() const {
    return std::ranges::views::iota(vertex_begin(), vertex_end());
  }

  auto local_num_vertices() const { return vertex_end() - vertex_begin(); }

  auto global_num_vertices() const { return _vertex_distribution.back(); }

  auto neighbors(VertexId v) const {
    auto begin = _xadj[v - vertex_begin()];
    auto end = _xadj[v - vertex_begin() + 1];
    std::span span{_adjncy};
    span = span.subspan(begin, end - begin);
    return span;
  }

  std::vector<int> get_comm_partners() const {
    std::unordered_set<int> comm_partners_set;
    std::vector<int> comm_partners;
    for (auto v : vertices()) {
      for (auto [_, rank] : neighbors(v)) {
        comm_partners_set.insert(rank);
      }
    }
    for (const auto &v : comm_partners_set) {
      comm_partners.emplace_back(v);
    }
    std::sort(comm_partners.begin(), comm_partners.end());
    return comm_partners;
  }

 private:
  size_t _rank;
  std::vector<VertexId> _xadj;
  std::vector<Edge> _adjncy;
  std::vector<VertexId> _vertex_distribution;
  std::vector<int> _home_rank;
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
  gen.discard(10);  // adavance internal state
  std::uniform_int_distribution<graph::VertexId> vertex_dist(
      0, g.global_num_vertices());
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
void graph_ping_pong(const graph::Graph &g, MPI_Comm comm) {
  using namespace graph;

  Frontier distributed_frontier{comm};
  [[maybe_unused]] volatile bool b = false;
  for (size_t i = 0; i < 10; ++i) {
    for (const auto &v : g.vertices()) {
      for (const auto &[u, rank] : g.neighbors(v)) {
        distributed_frontier.add_vertex(u, rank);
      }
    }

    kamping::measurements::timer().synchronize_and_start("alltoall");
    auto result = distributed_frontier.exchange();
    kamping::measurements::timer().stop_and_append();
    b = result.second;
  }
}

}  // namespace graph
