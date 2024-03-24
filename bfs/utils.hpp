#pragma once

#include <kagen.h>
#include <kagen/tools/random_permutation.h>

#include <iomanip>
#include <kamping/collectives/allgather.hpp>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/barrier.hpp>
#include <kamping/communicator.hpp>
#include <kamping/utils/flatten.hpp>

#include "common.hpp"

namespace graph {

inline int get_target_rank(VertexId v, std::size_t num_vertices,
                           std::size_t num_ranks) {
  std::size_t chunk_size = static_cast<std::size_t>(std::ceil(
      static_cast<double>(num_vertices) / static_cast<double>(num_ranks)));
  return static_cast<int>(v / chunk_size);
}

inline size_t get_first_vertex(size_t rank, std::size_t num_vertices,
                               std::size_t num_ranks) {
  return rank *
         static_cast<std::size_t>(std::ceil(static_cast<double>(num_vertices) /
                                            static_cast<double>(num_ranks)));
}

template <typename Permutation>
std::vector<int> compute_send_counts(const Permutation& permutation,
                                     const Graph& graph) {
  kamping::Communicator comm;
  const size_t n = graph.global_num_vertices();
  std::vector<int> send_counts(comm.size(), 0);
  for (VertexId i = 0; i < graph.local_num_vertices(); ++i) {
    VertexId global_id = i + graph.vertex_begin();
    VertexId permuted_id = permutation.f(global_id);
    const int target_rank = get_target_rank(permuted_id, n, comm.size());
    send_counts[target_rank] +=
        static_cast<int>(graph.neighbors(global_id).size());
  }
  return send_counts;
}
//
struct VHead {
  VertexId v;
  Edge head;
  bool operator<(const VHead& other) const { return v < other.v; }
};

template <typename Permutation>
auto exchange_edges(const Permutation& permutation, const Graph& graph) {
  kamping::Communicator<> comm;
  const size_t n = graph.global_num_vertices();

  std::unordered_map<int, std::vector<VHead>> send_buffer;
  for (VertexId i = 0; i < graph.local_num_vertices(); ++i) {
    VertexId global_id = i + graph.vertex_begin();
    VertexId permuted_id = permutation.f(global_id);
    const int target_rank = get_target_rank(permuted_id, n, comm.size());
    for (auto [u, rank] : graph.neighbors(global_id)) {
      VHead v_head;
      v_head.v = permuted_id;
      v_head.head.u = permutation.f(u);
      v_head.head.rank = get_target_rank(v_head.head.u, n, comm.size());
      send_buffer[target_rank].push_back(v_head);
    }
  }
  return kamping::with_flattened(send_buffer, comm.size())
      .call([&](auto... flattened) {
        return comm.alltoallv(std::move(flattened)...);
      });
}

template <typename Permutation>
auto permute_graph(const Permutation& permutation, const Graph& graph) {
  // debug_print(graph, " graph");
  kamping::Communicator<> comm;
  auto vheads = exchange_edges(permutation, graph);
  std::sort(vheads.begin(), vheads.end());

  //// debug_print(vheads, "vheads");
  std::vector<Edge> edges;
  edges.push_back(vheads.front().head);
  std::size_t cur_offset = 0;
  std::vector<VertexId> vertex_offsets{cur_offset};
  ++cur_offset;
  for (std::size_t i = 1; i < vheads.size(); ++i) {
    auto [v_prev, _] = vheads[i - 1];
    const auto [v, head] = vheads[i];
    while (v_prev != v) {
      vertex_offsets.push_back(cur_offset);
      ++v_prev;
    }
    edges.push_back(head);
    ++cur_offset;
  }
  vertex_offsets.push_back(cur_offset);
  const VertexId vertex_begin =
      get_first_vertex(comm.rank(), graph.global_num_vertices(), comm.size());
  std::vector<VertexId> vertex_distribution =
      comm.allgather(kamping::send_buf(vertex_begin));
  vertex_distribution.emplace_back(graph.global_num_vertices());
  Graph permuted_graph{std::move(vertex_offsets), std::move(edges),
                       std::move(vertex_distribution), comm};
  return permuted_graph;
}

inline auto construct_vertex_permutation(size_t num_global_vertices,
                                         size_t seed) {
  return kagen::random_permutation::FeistelPseudoRandomPermutation::
      buildPermutation(num_global_vertices - 1, seed);
}

template <typename Permutation>
void print_permutation_on_root(const Permutation& permutation, std::size_t n) {
  if (kamping::comm_world().rank() == 0) {
    const int width = static_cast<int>(std::log10(n) + 1);
    for (std::size_t i = 0; i < n; ++i) {
      std::cout << std::setw(width) << i << " ";
    }
    std::cout << "\n";
    for (std::size_t i = 0; i < n; ++i) {
      std::cout << std::setw(width) << permutation.f(i) << " ";
    }
    std::cout << "\n";
    for (std::size_t i = 0; i < n; ++i) {
      std::cout << std::setw(width) << permutation.finv(i) << " ";
    }
    std::cout << "\n" << std::flush;
  }
}

inline VertexId permute_vertex(size_t num_global_vertices, size_t seed,
                           VertexId vertex) {
  auto permutation = construct_vertex_permutation(num_global_vertices, seed);
  return permutation.f(vertex);
}

inline auto permute(const Graph& graph, size_t seed) {
  auto permutation =
      construct_vertex_permutation(graph.global_num_vertices(), seed);
  return permute_graph(permutation, graph);
}
}  // namespace graph
