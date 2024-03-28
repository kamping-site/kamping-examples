#pragma once
#include <random>

#include "sorting/common.hpp"
#include "kamping/mpi_datatype.hpp"

namespace mpi_new {
namespace helper = kamping;
template <typename T>
void sort(MPI_Comm comm, std::vector<T> &data, size_t seed) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  const size_t oversampling_ratio = 16 * std::log2(size) + 1;
  std::vector<T> local_samples(oversampling_ratio);
  std::sample(data.begin(), data.end(), local_samples.begin(),
              oversampling_ratio, std::mt19937{seed});
  std::vector<T> global_samples(local_samples.size() * size);
  MPI_Allgather(local_samples.data(), local_samples.size(),
                helper::mpi_datatype<T>(), global_samples.data(),
                local_samples.size(), helper::mpi_datatype<T>(), comm);
  pick_splitters(size - 1, oversampling_ratio, global_samples);
  auto buckets = build_buckets(data, global_samples);
  std::vector<int> sCounts, sDispls, rCounts(size), rDispls(size);
  int send_pos = 0;
  for (auto &bucket : buckets) {
    data.insert(data.end(), bucket.begin(), bucket.end());
    sCounts.push_back(bucket.size());
    sDispls.push_back(send_pos);
    send_pos += bucket.size();
  }
  MPI_Alltoall(sCounts.data(), 1, MPI_INT, rCounts.data(), 1, MPI_INT, comm);
  // exclusive prefix sum of recv displacements
  std::exclusive_scan(rCounts.begin(), rCounts.end(), rDispls.begin(), 0);
  std::vector<T> rData(rDispls.back() + rCounts.back());
  MPI_Alltoallv(data.data(), sCounts.data(), sDispls.data(),
                helper::mpi_datatype<T>(), rData.data(), rCounts.data(),
                rDispls.data(), helper::mpi_datatype<T>(), comm);
  std::sort(rData.begin(), rData.end());
  rData.swap(data);
}
}  // namespace mpi_new
