#pragma once
#include <kamping/collectives/allgather.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/communicator.hpp>
#include <kamping/utils/flatten.hpp>
#include <random>

#include "sorting/common.hpp"

namespace kamping {
//> START SORTING KAMPING
template <typename T>
void sort(MPI_Comm comm_, std::vector<T> &data, size_t seed) {
  using namespace kamping;
  kamping::Communicator comm(comm_);
  const size_t oversampling_ratio = 16 * std::log2(comm.size()) + 1;
  std::vector<T> local_samples(oversampling_ratio);
  std::sample(data.begin(), data.end(), local_samples.begin(),
              oversampling_ratio, std::mt19937{seed});
  std::vector<T> global_samples(local_samples.size() * comm.size());
  comm.allgather(send_buf(local_samples), recv_buf(global_samples));
  pick_splitters(comm.size() - 1, oversampling_ratio, global_samples);
  auto buckets = build_buckets(data, global_samples);
  std::vector<int> sCounts, sDispls, rCounts(comm.size()), rDispls(comm.size());
  int send_pos = 0;
  for (auto &bucket : buckets) {
    data.insert(data.end(), bucket.begin(), bucket.end());
    sCounts.push_back(bucket.size());
    sDispls.push_back(send_pos);
    send_pos += bucket.size();
  }
  comm.alltoall(kamping::send_buf(sCounts), kamping::recv_buf(rCounts));
  // exclusive prefix sum of recv displacements
  std::exclusive_scan(rCounts.begin(), rCounts.end(), rDispls.begin(), 0);
  std::vector<T> rData(rDispls.back() + rCounts.back());
  comm.alltoallv(send_buf(data), send_counts(sCounts), send_displs(sDispls),
                 recv_buf(rData), recv_counts(rCounts), recv_displs(rDispls));
  std::sort(rData.begin(), rData.end());
  rData.swap(data);
}
//> END
}  // namespace kamping
