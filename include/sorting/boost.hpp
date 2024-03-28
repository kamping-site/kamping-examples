#pragma once
#include <boost/mpi.hpp>
#include <random>

#include "./common.hpp"
namespace boost {
template <typename T>
void sort(MPI_Comm comm_, std::vector<T> &data, size_t seed) {
  boost::mpi::communicator comm(comm_, boost::mpi::comm_attach);
  const size_t oversampling_ratio = 16 * std::log2(comm.size()) + 1;
  std::vector<T> local_samples(oversampling_ratio);
  std::sample(data.begin(), data.end(), local_samples.begin(),
              oversampling_ratio, std::mt19937{seed});
  std::vector<T> global_samples;
  boost::mpi::all_gather(comm, local_samples.data(), local_samples.size(),
                         global_samples);
  pick_splitters(comm.size() - 1, oversampling_ratio, global_samples);
  auto buckets = build_buckets(data, global_samples);
  std::vector<int> sCounts, sDispls;
  sDispls.push_back(0);
  for (auto &bucket : buckets) {
    data.insert(data.end(), bucket.begin(), bucket.end());
    sCounts.push_back(bucket.size());
    sDispls.push_back(bucket.size() + sDispls.back());
  }
  std::vector<int> rCounts(comm.size());
  boost::mpi::all_to_all(comm, sCounts, rCounts);
  std::vector<int> rDispls(comm.size());
  std::exclusive_scan(rCounts.begin(), rCounts.end(), rDispls.begin(), 0);
  std::vector<T> rData(rCounts.back() + rDispls.back());  // data exchange
  // Boost.MPI does not support alltoallv
  MPI_Alltoallv(data.data(), sCounts.data(), sDispls.data(),
                boost::mpi::get_mpi_datatype<T>(), rData.data(), rCounts.data(),
                rDispls.data(), boost::mpi::get_mpi_datatype<T>(), comm);
  std::sort(rData.begin(), rData.end());
  rData.swap(data);
}
}  // namespace boost
