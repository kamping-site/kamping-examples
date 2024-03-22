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
    graph::VertexBuffer send_buffer;
    mpl::layouts<graph::VertexId> send_layouts;
    std::vector<int> send_counts(static_cast<size_t>(_comm.size()));
    int send_displ = 0;
    for (size_t rank = 0; rank < static_cast<size_t>(_comm.size()); ++rank) {
      auto it = _data.find(static_cast<int>(rank));
      if (it == _data.end()) {
        send_counts[rank] = 0;
        send_layouts.push_back(
            mpl::indexed_layout<graph::VertexId>({{0, send_displ}}));
        continue;
      }
      auto &local_data = it->second;
      send_counts[rank] = static_cast<int>(local_data.size());
      send_layouts.push_back(mpl::indexed_layout<graph::VertexId>(
          {{send_counts[rank], send_displ}}));
      send_buffer.insert(send_buffer.end(), local_data.begin(),
                         local_data.end());
      send_displ += send_counts[rank];
    }
    _data.clear();
    std::vector<int> recv_counts(static_cast<size_t>(_comm.size()));
    _comm.alltoall(send_counts.data(), recv_counts.data());
    int recv_displ = 0;
    mpl::layouts<graph::VertexId> recv_layouts;
    for (size_t i = 0; i < static_cast<size_t>(_comm.size()); ++i) {
      recv_layouts.push_back(
          mpl::indexed_layout<graph::VertexId>({{recv_counts[i], recv_displ}}));
      recv_displ += recv_counts[i];
    }
    std::vector<graph::VertexId> new_frontier(static_cast<size_t>(recv_displ));
    _comm.alltoallv(send_buffer.data(), send_layouts, new_frontier.data(),
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
