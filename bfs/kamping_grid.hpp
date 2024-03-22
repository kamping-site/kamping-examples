#pragma once

#include "common.hpp"
#include "kamping/plugin/alltoall_grid.hpp"
#include "kamping/utils/flatten.hpp"

namespace bfs_kamping_grid {
using namespace kamping;
class BFSFrontier final : public graph::BFSFrontier {
 public:
  BFSFrontier(MPI_Comm comm)
      : _comm{comm}, _grid_comm{_comm.make_grid_communicator()} {}
  std::pair<graph::VertexBuffer, bool> exchange() override {
    using namespace plugin::sparse_alltoall;
    if (is_empty()) {
      return std::make_pair(graph::VertexBuffer{}, true);
    }
    auto new_frontier =
        with_flattened(_data, _comm.size()).call([&](auto... flattened) {
          _data.clear();
          return _grid_comm.alltoallv(std::move(flattened)...);
        });
    return std::make_pair(std::move(new_frontier), false);
  }

  bool is_empty() const {
    return _comm.allreduce_single(send_buf(_data.empty()),
                                  op(std::logical_and<>{}));
  }

 private:
  kamping::Communicator<std::vector, plugin::GridCommunicator> _comm;
  plugin::grid::GridCommunicator<std::vector> _grid_comm;
};
}  // namespace kamping_grid
