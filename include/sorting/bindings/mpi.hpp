#pragma once
#include <random>

#include "kamping/mpi_datatype.hpp"
#include "sorting/common.hpp"

namespace mpi {
namespace helper = kamping;
template <typename T>
void sort(MPI_Comm comm, std::vector<T> &data, size_t seed) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  std::mt19937 eng(seed);
  const size_t oversampling_ratio = 16 * std::log2(size) + 1;
  std::vector<T> local_samples(oversampling_ratio);
  std::sample(data.begin(), data.end(), local_samples.begin(),
              oversampling_ratio, std::mt19937{seed});
  std::vector<T> global_samples(local_samples.size() * size);
  MPI_Allgather(local_samples.data(), local_samples.size(),
                helper::mpi_datatype<T>(), global_samples.data(),
                static_cast<int>(local_samples.size()),
                helper::mpi_datatype<T>(), comm);
  pick_splitters(size - 1, oversampling_ratio, global_samples);
  auto buckets = build_buckets(data, global_samples);
  std::vector<int> sCounts, sDispls, rCounts(size), rDispls(size + 1);
  sDispls.push_back(0);
  for (auto &bucket : buckets) {
    data.insert(data.end(), bucket.begin(), bucket.end());
    sCounts.push_back(bucket.size());
    sDispls.push_back(bucket.size() + sDispls.back());
  }
  MPI_Alltoall(sCounts.data(), 1, MPI_INT, rCounts.data(), 1, MPI_INT, comm);
  // exclusive prefix sum of recv displacements
  rDispls[0] = 0;
  for (size_t i = 1; i <= size; i++) {
    rDispls[i] = rCounts[i - 1] + rDispls[i - 1];
  }
  std::vector<T> rData(rDispls.back());  // data exchange
  MPI_Alltoallv(data.data(), sCounts.data(), sDispls.data(),
                helper::mpi_datatype<T>(), rData.data(), rCounts.data(),
                rDispls.data(), helper::mpi_datatype<T>(), comm);
  std::sort(rData.begin(), rData.end());
  rData.swap(data);
}
}  // namespace mpi
