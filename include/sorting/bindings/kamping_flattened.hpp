#pragma once
#include <kamping/collectives/allgather.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/communicator.hpp>
#include <kamping/utils/flatten.hpp>
#include <random>

#include "sorting/common.hpp"

namespace kamping_flattened {
//> START SORTING KAMPING_FLATTENED
template <typename T>
void sort(MPI_Comm comm_, std::vector<T> &data, size_t seed) {
  using namespace kamping;
  Communicator<> comm(comm_);
  const size_t oversampling_ratio = 16 * std::log2(comm.size()) + 1;
  std::vector<T> local_samples(oversampling_ratio);
  std::sample(data.begin(), data.end(), local_samples.begin(),
              oversampling_ratio, std::mt19937{seed});
  auto global_samples = comm.allgather(send_buf(local_samples));
  pick_splitters(comm.size() - 1, oversampling_ratio, global_samples);
  auto buckets = build_buckets(data, global_samples);
  data = with_flattened(buckets, comm.size()).call([&](auto... flattened) {
    return comm.alltoallv(std::move(flattened)...);
  });
  std::sort(data.begin(), data.end());
}
//> END
}  // namespace kamping_flattened
