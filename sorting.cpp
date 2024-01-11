#include <algorithm>
#include <boost/mpi.hpp>
#include <iostream>
#include <kamping/collectives/allgather.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/gather.hpp>
#include <kamping/communicator.hpp>
#include <mpi.h>
#define MPL_DEBUG 1
#include <mpi/all.hpp>
#include <mpl/mpl.hpp>
#include <random>
#include <sstream>
#include <vector>

bool globally_sorted(MPI_Comm comm, std::vector<int> const &data,
                     std::vector<int> &original_data) {
  kamping::Communicator kamping_comm(comm);
  auto global_data =
      kamping_comm.gatherv(kamping::send_buf(data)).extract_recv_buffer();
  auto global_data_original =
      kamping_comm.gatherv(kamping::send_buf(original_data))
          .extract_recv_buffer();
  std::sort(global_data_original.begin(), global_data_original.end());
  return global_data_original == global_data;
  // std::is_sorted(global_data.begin(), global_data.end());
}

void parallelSort(MPI_Comm comm, std::vector<int> &data) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  std::mt19937 eng(std::random_device{}());
  std::uniform_int_distribution<int> dist(0, data.size() - 1);
  const size_t oversampling_ratio = 16 * std::log2(size) + 1;
  std::vector<int> local_samples(oversampling_ratio);
  for (size_t i = 0; i < oversampling_ratio; i++) {
    local_samples[i] = data[dist(eng)];
  }
  std::vector<int> global_samples(local_samples.size() * size);
  MPI_Allgather(local_samples.data(), local_samples.size(), MPI_INT,
                global_samples.data(), local_samples.size(), MPI_INT, comm);

  std::sort(global_samples.begin(), global_samples.end());
  for (size_t i = 0; i < size - 1; i++) {
    global_samples[i] = global_samples[oversampling_ratio * (i + 1)];
  }
  global_samples.resize(size - 1);
  std::vector<std::vector<int>> buckets(size);
  for (auto &element : data) {
    const auto bound =
        std::upper_bound(global_samples.begin(), global_samples.end(), element);
    buckets[bound - global_samples.begin()].push_back(element);
  }
  data.clear();
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
  for (int i = 1; i <= size; i++) {
    rDispls[i] = rCounts[i - 1] + rDispls[i - 1];
  }
  std::vector<int> rData(rDispls.back()); // data exchange
  MPI_Alltoallv(data.data(), sCounts.data(), sDispls.data(), MPI_INT,
                rData.data(), rCounts.data(), rDispls.data(), MPI_INT, comm);
  std::sort(rData.begin(), rData.end());
  rData.swap(data);
}

void parallelSortKaMPIng(MPI_Comm comm_, std::vector<int> &data) {
  using namespace kamping;
  Communicator<> comm(comm_);
  std::mt19937 eng(std::random_device{}());
  std::uniform_int_distribution<int> dist(0, data.size() - 1);
  const size_t oversampling_ratio = 16 * std::log2(comm.size()) + 1;
  std::vector<int> local_samples(oversampling_ratio);
  for (size_t i = 0; i < oversampling_ratio; i++) {
    local_samples[i] = data[dist(eng)];
  }
  auto global_samples =
      comm.allgather(send_buf(local_samples)).extract_recv_buffer();

  std::sort(global_samples.begin(), global_samples.end());
  for (size_t i = 0; i < comm.size() - 1; i++) {
    global_samples[i] = global_samples[oversampling_ratio * (i + 1)];
  }
  global_samples.resize(comm.size() - 1);
  std::vector<std::vector<int>> buckets(comm.size());
  for (auto &element : data) {
    const auto bound =
        std::upper_bound(global_samples.begin(), global_samples.end(), element);
    buckets[bound - global_samples.begin()].push_back(element);
  }
  data.clear();
  std::vector<int> scounts;
  for (auto &bucket : buckets) {
    data.insert(data.end(), bucket.begin(), bucket.end());
    scounts.push_back(bucket.size());
  }
  data = comm.alltoallv(send_buf(std::move(data)), send_counts(scounts))
             .extract_recv_buffer();
  std::sort(data.begin(), data.end());
}

void parallelSortMPL(MPI_Comm comm_, std::vector<int> &data) {
  mpl::communicator comm{mpl::environment::comm_world()};
  std::mt19937 eng(std::random_device{}());
  std::uniform_int_distribution<int> dist(0, data.size() - 1);
  const size_t oversampling_ratio = 16 * std::log2(comm.size()) + 1;
  std::vector<int> local_samples(oversampling_ratio);
  for (size_t i = 0; i < oversampling_ratio; i++) {
    local_samples[i] = data[dist(eng)];
  }
  std::vector<int> global_samples(local_samples.size() * comm.size());
  auto send_layout = mpl::vector_layout<int>(local_samples.size());
  auto recv_layout = mpl::vector_layout<int>(local_samples.size());
  comm.allgather(local_samples.data(), send_layout, global_samples.data(),
                 recv_layout);
  std::sort(global_samples.begin(), global_samples.end());
  for (size_t i = 0; i < comm.size() - 1; i++) {
    global_samples[i] = global_samples[oversampling_ratio * (i + 1)];
  }
  global_samples.resize(comm.size() - 1);
  std::vector<std::vector<int>> buckets(comm.size());
  for (auto &element : data) {
    const auto bound =
        std::upper_bound(global_samples.begin(), global_samples.end(), element);
    buckets[bound - global_samples.begin()].push_back(element);
  }
  data.clear();
  mpl::layouts<int> send_layouts;
  std::vector<int> send_counts;
  int send_pos = 0;
  for (auto &bucket : buckets) {
    data.insert(data.end(), bucket.begin(), bucket.end());
    send_counts.push_back(bucket.size());
    send_layouts.push_back(mpl::indexed_layout<int>({{bucket.size(), send_pos}}));
    send_pos += bucket.size();
  }
  std::vector<int> recv_counts(comm.size());
  comm.alltoall(send_counts.data(), recv_counts.data());
  int recv_pos = 0;
  mpl::layouts<int> recv_layouts;
  for (int i = 0; i < comm.size(); i++) {
    recv_layouts.push_back(mpl::indexed_layout<int>({{recv_counts[i], recv_pos}}));
    recv_pos += recv_counts[i];
  }
  std::vector<int> recv_data(recv_pos); // data exchange
  comm.alltoallv(data.data(), send_layouts, recv_data.data(), recv_layouts);
  std::sort(recv_data.begin(), recv_data.end());
  recv_data.swap(data);
}

void parallelSortBoostMPI(MPI_Comm comm_, std::vector<int> &data) {
  boost::mpi::communicator comm(comm_, boost::mpi::comm_attach);
  std::mt19937 eng(std::random_device{}());
  std::uniform_int_distribution<int> dist(0, data.size() - 1);
  const size_t oversampling_ratio = 16 * std::log2(comm.size()) + 1;
  std::vector<int> local_samples(oversampling_ratio);
  for (size_t i = 0; i < oversampling_ratio; i++) {
    local_samples[i] = data[dist(eng)];
  }
  std::vector<int> global_samples;
  boost::mpi::all_gather(comm, local_samples.data(), local_samples.size(),
                         global_samples);

  std::sort(global_samples.begin(), global_samples.end());
  for (size_t i = 0; i < comm.size() - 1; i++) {
    global_samples[i] = global_samples[oversampling_ratio * (i + 1)];
  }
  global_samples.resize(comm.size() - 1);
  std::vector<std::vector<int>> buckets(comm.size());
  for (auto &element : data) {
    const auto bound =
        std::upper_bound(global_samples.begin(), global_samples.end(), element);
    buckets[bound - global_samples.begin()].push_back(element);
  }
  data.clear();
  std::vector<int> sCounts, sDispls;
  sDispls.push_back(0);
  for (auto &bucket : buckets) {
    data.insert(data.end(), bucket.begin(), bucket.end());
    sCounts.push_back(bucket.size());
    sDispls.push_back(bucket.size() + sDispls.back());
  }

  std::vector<int> rCounts(comm.size());
  boost::mpi::all_to_all(comm, sCounts, rCounts);

  // exclusive prefix sum of recv displacements
  std::vector<int> rDispls(comm.size() + 1);
  rDispls[0] = 0;
  for (int i = 1; i <= comm.size(); i++) {
    rDispls[i] = rCounts[i - 1] + rDispls[i - 1];
  }
  std::vector<int> rData(rDispls.back()); // data exchange
  // Boost.MPI does not support alltoallv
  MPI_Alltoallv(data.data(), sCounts.data(), sDispls.data(), MPI_INT,
                rData.data(), rCounts.data(), rDispls.data(), MPI_INT, comm);
  std::sort(rData.begin(), rData.end());
  rData.swap(data);
}

void parallelSortRWTHMPI(MPI_Comm comm_, std::vector<int> &data) {
  mpi::communicator comm(comm_);
  std::mt19937 eng(std::random_device{}());
  std::uniform_int_distribution<int> dist(0, data.size() - 1);
  const size_t oversampling_ratio = 16 * std::log2(comm.size()) + 1;
  std::vector<int> local_samples(oversampling_ratio);
  for (size_t i = 0; i < oversampling_ratio; i++) {
    local_samples[i] = data[dist(eng)];
  }
  std::vector<int> global_samples(local_samples.size() * comm.size());
  comm.all_gather(local_samples, global_samples);

  std::sort(global_samples.begin(), global_samples.end());
  for (size_t i = 0; i < comm.size() - 1; i++) {
    global_samples[i] = global_samples[oversampling_ratio * (i + 1)];
  }
  global_samples.resize(comm.size() - 1);
  std::vector<std::vector<int>> buckets(comm.size());
  for (auto &element : data) {
    const auto bound =
        std::upper_bound(global_samples.begin(), global_samples.end(), element);
    buckets[bound - global_samples.begin()].push_back(element);
  }
  data.clear();
  std::vector<int> sCounts;
  for (auto &bucket : buckets) {
    data.insert(data.end(), bucket.begin(), bucket.end());
    sCounts.push_back(bucket.size());
  }
  std::vector<int> rData;
  comm.all_to_all_varying(data, sCounts, rData, true);
  std::sort(rData.begin(), rData.end());
  rData.swap(data);
}

int main(int argc, char *argv[]) {
  // MPI_Init(&argc, &argv);
  mpl::environment::comm_world();

  std::string algorithm{argv[1]};
  std::stringstream n_str(argv[2]);
  size_t n;
  n_str >> n;
  std::mt19937 eng(std::random_device{}());
  std::uniform_int_distribution<int> dist(0, 100);
  std::vector<int> data(n);
  auto gen = [&] { return dist(eng); };
  std::generate(data.begin(), data.end(), gen);
  auto original_data = data;
  if (algorithm == "mpi") {
    parallelSort(MPI_COMM_WORLD, data);
  } else if (algorithm == "kamping") {
    parallelSortKaMPIng(MPI_COMM_WORLD, data);
  } else if (algorithm == "boost") {
    parallelSortBoostMPI(MPI_COMM_WORLD, data);
  } else if (algorithm == "rwth") {
    parallelSortRWTHMPI(MPI_COMM_WORLD, data);
  } else if (algorithm == "mpl") {
    parallelSortMPL(MPI_COMM_WORLD, data);
  } else {
    throw std::runtime_error("unsupported algorithm");
  }
  std::cout << std::boolalpha << "sorted: "
            << globally_sorted(MPI_COMM_WORLD, data, original_data)
            << std::endl;
  // MPI_Finalize();
  return 0;
}
