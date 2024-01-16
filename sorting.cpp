#include <CLI/CLI.hpp>
#include <algorithm>
#include <boost/mpi.hpp>
#include <iostream>
#include <kamping/collectives/allgather.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/gather.hpp>
#include <kamping/communicator.hpp>
#include <kamping/measurements/printer.hpp>
#include <kamping/measurements/timer.hpp>
#include <mpi.h>
#include <mpi/all.hpp>
#include <mpl/mpl.hpp>
#include <random>
#include <sstream>
#include <vector>

template <typename T>
bool globally_sorted(MPI_Comm comm, std::vector<T> const &data,
                     std::vector<T> &original_data) {
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

template <typename T>
void pick_splitters(size_t num_splitters, size_t oversampling_ratio,
                    std::vector<T> &global_samples) {
  std::sort(global_samples.begin(), global_samples.end());
  for (size_t i = 0; i < num_splitters; i++) {
    global_samples[i] = global_samples[oversampling_ratio * (i + 1)];
  }
  global_samples.resize(num_splitters);
}

template <typename T>
auto build_buckets(std::vector<T> &data, std::vector<T> &splitters)
    -> std::vector<std::vector<T>> {
  std::vector<std::vector<T>> buckets(splitters.size() + 1);
  for (auto &element : data) {
    const auto bound =
        std::upper_bound(splitters.begin(), splitters.end(), element);
    buckets[bound - splitters.begin()].push_back(element);
  }
  data.clear();
  return buckets;
}

template <typename T>
void parallelSort(MPI_Comm comm, std::vector<T> &data, size_t seed) {
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
                kamping::mpi_datatype<T>(), global_samples.data(),
                local_samples.size(), kamping::mpi_datatype<T>(), comm);
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
  for (int i = 1; i <= size; i++) {
    rDispls[i] = rCounts[i - 1] + rDispls[i - 1];
  }
  std::vector<T> rData(rDispls.back()); // data exchange
  MPI_Alltoallv(data.data(), sCounts.data(), sDispls.data(),
                kamping::mpi_datatype<T>(), rData.data(), rCounts.data(),
                rDispls.data(), kamping::mpi_datatype<T>(), comm);
  std::sort(rData.begin(), rData.end());
  rData.swap(data);
}
template <typename T>
void parallelSortImproved(MPI_Comm comm, std::vector<T> &data, size_t seed) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  const size_t oversampling_ratio = 16 * std::log2(size) + 1;
  std::vector<T> local_samples(oversampling_ratio);
  std::sample(data.begin(), data.end(), local_samples.begin(),
              oversampling_ratio, std::mt19937{seed});
  std::vector<T> global_samples(local_samples.size() * size);
  MPI_Allgather(local_samples.data(), local_samples.size(),
                kamping::mpi_datatype<T>(), global_samples.data(),
                local_samples.size(), kamping::mpi_datatype<T>(), comm);

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
                kamping::mpi_datatype<T>(), rData.data(), rCounts.data(),
                rDispls.data(), kamping::mpi_datatype<T>(), comm);
  std::sort(rData.begin(), rData.end());
  rData.swap(data);
}

template <typename T>
void parallelSortKaMPIng(MPI_Comm comm_, std::vector<T> &data, size_t seed) {
  using namespace kamping;
  Communicator<> comm(comm_);
  const size_t oversampling_ratio = 16 * std::log2(comm.size()) + 1;
  std::vector<T> local_samples(oversampling_ratio);
  std::sample(data.begin(), data.end(), local_samples.begin(),
              oversampling_ratio, std::mt19937{seed});
  auto global_samples =
      comm.allgather(send_buf(local_samples)).extract_recv_buffer();
  pick_splitters(comm.size() - 1, oversampling_ratio, global_samples);
  auto buckets = build_buckets(data, global_samples);
  std::vector<int> scounts;
  for (auto &bucket : buckets) {
    data.insert(data.end(), bucket.begin(), bucket.end());
    scounts.push_back(bucket.size());
  }
  data = comm.alltoallv(send_buf(std::move(data)), send_counts(scounts))
             .extract_recv_buffer();
  std::sort(data.begin(), data.end());
}

template <typename T>
void parallelSortMPL(MPI_Comm comm_, std::vector<T> &data, size_t seed) {
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
  for (int i = 0; i < comm.size(); i++) {
    recv_layouts.push_back(
        mpl::indexed_layout<T>({{recv_counts[i], recv_pos}}));
    recv_pos += recv_counts[i];
  }
  std::vector<T> recv_data(recv_pos); // data exchange
  comm.alltoallv(data.data(), send_layouts, recv_data.data(), recv_layouts);
  std::sort(recv_data.begin(), recv_data.end());
  recv_data.swap(data);
}

template <typename T>
void parallelSortBoostMPI(MPI_Comm comm_, std::vector<T> &data, size_t seed) {
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

  // exclusive prefix sum of recv displacements
  std::vector<int> rDispls(comm.size() + 1);
  rDispls[0] = 0;
  for (int i = 1; i <= comm.size(); i++) {
    rDispls[i] = rCounts[i - 1] + rDispls[i - 1];
  }
  std::vector<T> rData(rDispls.back()); // data exchange
  // Boost.MPI does not support alltoallv
  MPI_Alltoallv(data.data(), sCounts.data(), sDispls.data(),
                boost::mpi::get_mpi_datatype<T>(), rData.data(), rCounts.data(),
                rDispls.data(), boost::mpi::get_mpi_datatype<T>(), comm);
  std::sort(rData.begin(), rData.end());
  rData.swap(data);
}

template <typename T>
void parallelSortRWTHMPI(MPI_Comm comm_, std::vector<T> &data, size_t seed) {
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

template <typename T>
auto generate_data(size_t n_local, size_t seed) -> std::vector<T> {
  std::mt19937 eng(seed);
  std::uniform_int_distribution<T> dist(0, std::numeric_limits<T>::max());
  std::vector<T> data(n_local);
  auto gen = [&] { return dist(eng); };
  std::generate(data.begin(), data.end(), gen);
  return data;
}

void log_results(std::string const &json_output_path,
                 std::string const &algorithm, size_t n_local, size_t seed, bool correct) {
  std::unique_ptr<std::ostream> output_stream;
  if (json_output_path == "stdout") {
    output_stream = std::make_unique<std::ostream>(std::cout.rdbuf());
  } else {
    std::ofstream file_output(json_output_path);
    output_stream = std::make_unique<std::ofstream>(std::move(file_output));
  }
  if (mpl::environment::comm_world().rank() == 0) {
    *output_stream << "{\n";
  }
  kamping::measurements::timer().aggregate_and_print(
      kamping::measurements::SimpleJsonPrinter<>{*output_stream});
  if (mpl::environment::comm_world().rank() == 0) {
    *output_stream << ",\n";
    *output_stream << "\"info\": {\n";
    *output_stream << "  \"algorithm\": "
                   << "\"" << algorithm << "\",\n";
    *output_stream << "  \"p\": " << mpl::environment::comm_world().size()
                   << ",\n";
    *output_stream << "  \"n_local\": " << n_local << ",\n";
    *output_stream << "  \"seed\": " << seed << ",\n";
    *output_stream << "  \"correct\": " << std::boolalpha << correct << "\n";
    *output_stream << "}\n";
    *output_stream << "}";
  }
}

int main(int argc, char *argv[]) {
  mpl::environment::comm_world(); // this perform MPI_init, MPL has no other way
                                  // to do it and calls it implicitly when first
                                  // accessing a communicator
  CLI::App app{"Parallel sorting"};
  std::string algorithm;
  app.add_option("--algorithm", algorithm);
  size_t n_local;
  app.add_option("--n_local", n_local);
  size_t seed = 42;
  app.add_option("--seed", seed);
  size_t iterations = 1;
  app.add_option("--iterations", iterations);
  bool check;
  app.add_flag("--check", check);
  std::string json_output_path = "stdout";
  app.add_option("--json_output_path", json_output_path);
  CLI11_PARSE(app, argc, argv);

  using element_type = uint64_t;

  auto original_data = generate_data<element_type>(n_local, seed);
  bool correct = false;
  auto do_run = [&](auto &&algo) {
    if (check) {
      auto data = original_data;
      algo(MPI_COMM_WORLD, data, seed);
      correct = globally_sorted(MPI_COMM_WORLD, data, original_data);
    }
    for (size_t iteration = 0; iteration < iterations; iteration++) {
      auto data = original_data;
      kamping::measurements::timer().synchronize_and_start("total_time");
      algo(MPI_COMM_WORLD, data, seed);
      kamping::measurements::timer().stop_and_append();
    }
  };
  if (algorithm == "mpi") {
    do_run(parallelSort<element_type>);
  } else if (algorithm == "mpi_new") {
    do_run(parallelSortImproved<element_type>);
  } else if (algorithm == "kamping") {
    do_run(parallelSortKaMPIng<element_type>);
  } else if (algorithm == "boost") {
    do_run(parallelSortBoostMPI<element_type>);
  } else if (algorithm == "rwth") {
    do_run(parallelSortRWTHMPI<element_type>);
  } else if (algorithm == "mpl") {
    do_run(parallelSortMPL<element_type>);
  } else {
    throw std::runtime_error("unsupported algorithm");
  }
  log_results(json_output_path, algorithm, n_local, seed, correct);
  return 0;
}
