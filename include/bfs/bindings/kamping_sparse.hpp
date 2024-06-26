#pragma once

#include "bfs/common.hpp"
#include "kamping/collectives/allreduce.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/plugin/alltoall_sparse.hpp"
#include "kamping/utils/flatten.hpp"

namespace bfs_kamping_sparse {
//> START BFS KAMPING_SPARSE
using namespace kamping;
using namespace plugin::sparse_alltoall;
class BFSFrontier final : public graph::BFSFrontier {
 public:
  BFSFrontier(MPI_Comm comm) : _comm{comm} {}
  std::pair<graph::VertexBuffer, bool> exchange() override {
    if (is_empty()) {
      return std::make_pair(graph::VertexBuffer{}, true);
    }
    graph::VertexBuffer new_frontier;
    _comm.alltoallv_sparse(
        sparse_send_buf(_data), on_message([&](auto &probed_message) {
          auto prev_size = new_frontier.size();
          new_frontier.resize(new_frontier.size() +
                              probed_message.recv_count());
          Span message{new_frontier.begin() + prev_size, new_frontier.end()};
          probed_message.recv(recv_buf(message));
        }));
    _data.clear();
    return std::make_pair(std::move(new_frontier), false);
  }
  bool is_empty() const {
    return _comm.allreduce_single(send_buf(_data.empty()),
                                  op(std::logical_and<>{}));
  }

 private:
  kamping::Communicator<std::vector, plugin::SparseAlltoall> _comm;
};
//> END
}  // namespace bfs_kamping_sparse
