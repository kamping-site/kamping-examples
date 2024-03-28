#pragma once
#include <mpi.h>

#include <mpi/all.hpp>
#include <random>

#include "sorting/common.hpp"

namespace rwth {
template <typename T>
void sort(MPI_Comm comm_, std::vector<T> &data, size_t seed) {
  mpi::communicator comm(comm_);
  const size_t oversampling_ratio = 16 * std::log2(comm.size()) + 1;
  std::vector<T> local_samples(oversampling_ratio);
  std::sample(data.begin(), data.end(), local_samples.begin(),
              oversampling_ratio, std::mt19937{seed});
  std::vector<T> global_samples(local_samples.size() * comm.size());
  comm.all_gather(local_samples, global_samples);
  pick_splitters(comm.size() - 1, oversampling_ratio, global_samples);
  auto buckets = build_buckets(data, global_samples);
  std::vector<int> sCounts;
  for (auto &bucket : buckets) {
    data.insert(data.end(), bucket.begin(), bucket.end());
    sCounts.push_back(bucket.size());
  }
  std::vector<T> rData;
  comm.all_to_all_varying(data, sCounts, rData, true);
  std::sort(rData.begin(), rData.end());
  rData.swap(data);
}
}  // namespace rwth
