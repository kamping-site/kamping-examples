#pragma once
#include <mpl/mpl.hpp>
#include <random>

#include "sorting/common.hpp"

namespace mpl {
template <typename T>
void sort(MPI_Comm, std::vector<T> &data, size_t seed) {
  mpl::communicator comm{mpl::environment::comm_world()};
  const size_t oversampling_ratio = 16 * std::log2(comm.size()) + 1;
  std::vector<T> local_samples(oversampling_ratio);
  std::sample(data.begin(), data.end(), local_samples.begin(),
              oversampling_ratio, std::mt19937{seed});
  std::vector<T> global_samples(local_samples.size() * comm.size());
  auto send_layout = mpl::vector_layout<T>(local_samples.size());
  auto recv_layout = mpl::vector_layout<T>(local_samples.size());
  comm.allgather(local_samples.data(), send_layout, global_samples.data(),
                 recv_layout);
  pick_splitters(comm.size() - 1, oversampling_ratio, global_samples);
  auto buckets = build_buckets(data, global_samples);
  mpl::layouts<T> send_layouts;
  std::vector<int> send_counts;
  int send_pos = 0;
  for (auto &bucket : buckets) {
    data.insert(data.end(), bucket.begin(), bucket.end());
    send_counts.push_back(bucket.size());
    send_layouts.push_back(mpl::indexed_layout<T>({{bucket.size(), send_pos}}));
    send_pos += bucket.size();
  }
  std::vector<int> recv_counts(comm.size());
  comm.alltoall(send_counts.data(), recv_counts.data());
  int recv_pos = 0;
  mpl::layouts<T> recv_layouts;
  for (size_t i = 0; i < comm.size(); i++) {
    recv_layouts.push_back(
        mpl::indexed_layout<T>({{recv_counts[i], recv_pos}}));
    recv_pos += recv_counts[i];
  }
  std::vector<T> recv_data(recv_pos);  // data exchange
  comm.alltoallv(data.data(), send_layouts, recv_data.data(), recv_layouts);
  std::sort(recv_data.begin(), recv_data.end());
  recv_data.swap(data);
}
}  // namespace mpl
