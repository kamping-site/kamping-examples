#pragma once
#include <random>

#include "kamping/mpi_datatype.hpp"
#include "sorting/common.hpp"

namespace mpi_rdma {

namespace helper = kamping;
template <typename T>
std::vector<T> exchange_via_rdma(const std::vector<std::vector<T>> &buckets,
                                 const std::vector<int> &send_counts,
                                 size_t total_recv_count, int rank, int size,
                                 MPI_Comm comm) {
  std::vector<T> rData(total_recv_count);
  MPI_Win win, offset_win;
  size_t receive_offset;

  // create windows
  MPI_Win_create(&receive_offset, sizeof(size_t), sizeof(size_t), MPI_INFO_NULL,
                 comm, &offset_win);
  MPI_Win_create(rData.data(), rData.size() * sizeof(T), sizeof(T),
                 MPI_INFO_NULL, comm, &win);

  MPI_Win_lock(MPI_LOCK_SHARED, rank, MPI_MODE_NOCHECK, offset_win);
  receive_offset = 0u;
  MPI_Win_unlock(rank, offset_win);

  MPI_Barrier(comm);

  // retrieve write offset for local bucket
  size_t cur_write_offset = 0;
  {
    MPI_Win_lock(MPI_LOCK_SHARED, rank, MPI_MODE_NOCHECK, offset_win);
    size_t cur_send_count = static_cast<size_t>(send_counts[rank]);
    MPI_Fetch_and_op(&cur_send_count, &cur_write_offset,
                     helper::mpi_datatype<size_t>(), rank, 0, MPI_SUM,
                     offset_win);
    MPI_Win_unlock(rank, offset_win);
  }

  MPI_Win_lock_all(MPI_MODE_NOCHECK, win);

  // do not copy self bucket via RDMA but use copy instead using previously
  // obtained write offset
  std::copy_n(buckets[rank].begin(), send_counts[rank],
              rData.data() + cur_write_offset);

  MPI_Barrier(comm);
  for (int i = 1; i < size; ++i) {
    const int target_pe = (rank + i) % size;

    // fetch-and-add write offset for target_pe
    MPI_Win_lock(MPI_LOCK_SHARED, target_pe, MPI_MODE_NOCHECK, offset_win);
    size_t cur_send_count = static_cast<size_t>(send_counts[target_pe]);
    MPI_Fetch_and_op(&cur_send_count, &cur_write_offset,
                     helper::mpi_datatype<size_t>(), target_pe, 0, MPI_SUM,
                     offset_win);
    MPI_Win_unlock(target_pe, offset_win);

    // put data to target_pe
    MPI_Put(buckets[target_pe].data(), send_counts[target_pe],
            helper::mpi_datatype<T>(), target_pe, cur_write_offset,
            send_counts[target_pe], helper::mpi_datatype<T>(), win);
  }
  MPI_Win_unlock_all(win);
  MPI_Barrier(comm);

  MPI_Win_free(&offset_win);
  MPI_Win_free(&win);

  return rData;
}
//> START SORTING MPI
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
  std::vector<int> sCounts, rCounts(size);
  for (auto &bucket : buckets) {
    sCounts.push_back(bucket.size());
  }
  // replace with scatter-reduce
  MPI_Alltoall(sCounts.data(), 1, MPI_INT, rCounts.data(), 1, MPI_INT, comm);

  const size_t total_recv_counts =
      std::accumulate(rCounts.begin(), rCounts.end(), size_t{0});
  std::vector<T> rData =
      exchange_via_rdma(buckets, sCounts, total_recv_counts, rank, size, comm);
  std::sort(rData.begin(), rData.end());
  rData.swap(data);
}
//> END
}  // namespace mpi_rdma
