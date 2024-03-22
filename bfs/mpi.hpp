#pragma once

#include "common.hpp"
#include "kamping/mpi_datatype.hpp"

namespace helper = kamping;

namespace mpi {
class BFSFrontier final : public graph::BFSFrontier {
 public:
  BFSFrontier(MPI_Comm comm) : _comm{comm} {}
  std::pair<graph::VertexBuffer, bool> exchange() override {
    if (is_empty()) {
      return std::make_pair(graph::VertexBuffer{}, true);
    }
    int rank, size;
    MPI_Comm_rank(_comm, &rank);
    MPI_Comm_size(_comm, &size);
    std::vector<graph::VertexId> send_buffer;
    std::vector<int> send_counts(static_cast<size_t>(size));
    for (size_t i = 0; i < static_cast<size_t>(size); ++i) {
      auto it = _data.find(static_cast<int>(i));
      if (it == _data.end()) {
        send_counts[i] = 0;
        continue;
      }
      auto &local_data = it->second;
      send_buffer.insert(send_buffer.end(), local_data.begin(),
                         local_data.end());
      send_counts[i] = static_cast<int>(local_data.size());
      std::vector<graph::VertexId>{}.swap(local_data);
    }
    _data.clear();
    kamping::measurements::timer().start("alltoall");
    std::vector<int> recv_counts(static_cast<size_t>(size));
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
                 _comm);
    std::vector<int> send_displs(static_cast<size_t>(size));
    std::vector<int> recv_displs(static_cast<size_t>(size));
    std::exclusive_scan(send_counts.begin(), send_counts.end(),
                        send_displs.begin(), 0);
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(),
                        recv_displs.begin(), 0);
    const size_t num_recv_elems =
        static_cast<size_t>(recv_counts.back() + recv_displs.back());
    graph::VertexBuffer new_frontier(num_recv_elems);
    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(),
                  helper::mpi_datatype<graph::VertexId>(), new_frontier.data(),
                  recv_counts.data(), recv_displs.data(),
                  helper::mpi_datatype<graph::VertexId>(), _comm);
    return std::make_pair(std::move(new_frontier), false);
  }

  bool is_empty() const {
    bool result = _data.empty();
    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_CXX_BOOL, MPI_LAND, _comm);
    return result;
  }

 private:
  MPI_Comm _comm;
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
}  // namespace mpi
