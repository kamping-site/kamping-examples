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

#include "../mpi_spd_formatters.hpp"

namespace graph {

using VertexId = kagen::SInt;

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

/// @brief Represent the frontier in a distributed breadth-first search (BFS).
class BFSFrontier {
 public:
  void add_vertex(VertexId v, int rank) { _data[rank].push_back(v); }
  virtual std::vector<VertexId> exchange() = 0;
  virtual ~BFSFrontier(){};

 protected:
  std::unordered_map<int, std::vector<VertexId>>
      _data;  ///< map vertices of the frontier to their home rank
};

}  // namespace graph
