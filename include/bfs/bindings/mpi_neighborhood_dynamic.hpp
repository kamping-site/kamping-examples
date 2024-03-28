#pragma once

#include "bfs/common.hpp"
#include "kamping/mpi_datatype.hpp"

namespace helper = kamping;

namespace bfs_mpi_neighborhood_dynamic {
class BFSFrontier final : public graph::BFSFrontier {
 public:
  BFSFrontier(MPI_Comm comm, const std::vector<int>& comm_partners)
      : _comm{comm}, _comm_partners{comm_partners} {}
  std::pair<graph::VertexBuffer, bool> exchange() override {
    MPI_Comm comm_graph;
    MPI_Dist_graph_create_adjacent(
        _comm, static_cast<int>(_comm_partners.size()), _comm_partners.data(),
        MPI_UNWEIGHTED, static_cast<int>(_comm_partners.size()),
        _comm_partners.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false,
        &comm_graph);

    if (is_empty()) {
      return std::make_pair(graph::VertexBuffer{}, true);
    }
    int rank, size;
    MPI_Comm_rank(comm_graph, &rank);
    MPI_Comm_size(comm_graph, &size);
    graph::VertexBuffer data;
    std::vector<int> sCounts(_comm_partners.size(), 0);
    for (size_t i = 0; i < _comm_partners.size(); ++i) {
      auto it = _data.find(_comm_partners[i]);
      if (it == _data.end()) {
        continue;
      }
      auto& local_data = it->second;
      data.insert(data.end(), local_data.begin(), local_data.end());
      sCounts[i] = local_data.size();
    }
    _data.clear();
    std::vector<int> rCounts(_comm_partners.size());
    MPI_Neighbor_alltoall(sCounts.data(), 1, MPI_INT, rCounts.data(), 1,
                          MPI_INT, comm_graph);
    std::vector<int> sDispls(_comm_partners.size());
    std::vector<int> rDispls(_comm_partners.size());
    std::exclusive_scan(sCounts.begin(), sCounts.end(), sDispls.begin(), 0);
    std::exclusive_scan(rCounts.begin(), rCounts.end(), rDispls.begin(), 0);
    const size_t num_recv_elems =
        static_cast<size_t>(rCounts.back() + rDispls.back());
    graph::VertexBuffer new_frontier(num_recv_elems);
    MPI_Neighbor_alltoallv(data.data(), sCounts.data(), sDispls.data(),
                           helper::mpi_datatype<graph::VertexId>(),
                           new_frontier.data(), rCounts.data(), rDispls.data(),
                           helper::mpi_datatype<graph::VertexId>(), comm_graph);
    MPI_Comm_free(&comm_graph);
    return std::make_pair(std::move(new_frontier), false);
  }
  bool is_empty() const {
    bool result = _data.empty();
    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_CXX_BOOL, MPI_LAND, _comm);
    return result;
  }

 private:
  MPI_Comm _comm;
  std::vector<int> _comm_partners;
};
}  // namespace bfs_mpi_neighborhood_dynamic
