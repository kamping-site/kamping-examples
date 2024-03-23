#pragma once

#include "common.hpp"
#include "mpl/mpl.hpp"

namespace bfs_mpl {
class BFSFrontier final : public graph::BFSFrontier {
 public:
  BFSFrontier(MPI_Comm)
      : _comm{mpl::environment::comm_world()} {
  }  // MPL does not offer a c'tor accepting a MPI_Comm
  std::pair<graph::VertexBuffer, bool> exchange() override {
    if (is_empty()) {
      return std::make_pair(graph::VertexBuffer{}, true);
    }
    graph::VertexBuffer data;
    mpl::layouts<graph::VertexId> send_layouts;
    std::vector<int> sCounts(_comm.size());
    int send_displ = 0;
    for (size_t rank = 0; rank < _comm.size(); ++rank) {
      auto it = _data.find(rank);
      if (it == _data.end()) {
        sCounts[rank] = 0;
        send_layouts.push_back(
            mpl::indexed_layout<graph::VertexId>({{0, send_displ}}));
        continue;
      }
      auto &local_data = it->second;
      sCounts[rank] = local_data.size();
      send_layouts.push_back(mpl::indexed_layout<graph::VertexId>(
          {{sCounts[rank], send_displ}}));
      data.insert(data.end(), local_data.begin(),
                         local_data.end());
      send_displ += sCounts[rank];
    }
    _data.clear();
    std::vector<int> rCounts(_comm.size());
    _comm.alltoall(sCounts.data(), rCounts.data());
    int recv_displ = 0;
    mpl::layouts<graph::VertexId> recv_layouts;
    for (size_t i = 0; i < _comm.size(); ++i) {
      recv_layouts.push_back(
          mpl::indexed_layout<graph::VertexId>({{rCounts[i], recv_displ}}));
      recv_displ += rCounts[i];
    }
    std::vector<graph::VertexId> new_frontier(recv_displ);
    _comm.alltoallv(data.data(), send_layouts, new_frontier.data(),
                    recv_layouts);
    return std::make_pair(std::move(new_frontier), false);
  }

  bool is_empty() const {
    bool result;
    _comm.allreduce(std::logical_and<>{}, _data.empty(), result);
    return result;
  }

 private:
  mpl::communicator _comm;
};
}  // namespace mpl
