#if defined(KAMPING_EXAMPLES_USE_BOOST)
#include "detail/boost.hpp"
#endif
#include <mpi.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <iostream>
#include <kamping/collectives/gather.hpp>
#include <kamping/communicator.hpp>
#include <kamping/measurements/printer.hpp>
#include <kamping/measurements/timer.hpp>
#include <mpl/mpl.hpp>  // needed for initialization
#include <random>
#include <vector>

#include "sorting/common.hpp"
#include "sorting/kamping.hpp"
#include "sorting/mpi.hpp"
#include "sorting/mpi_new.hpp"
#include "sorting/mpl.hpp"
#include "sorting/rwth.hpp"

template <typename T>
bool globally_sorted(MPI_Comm comm, std::vector<T> const &data,
                     std::vector<T> &original_data) {
  kamping::Communicator kamping_comm(comm);
  auto global_data = kamping_comm.gatherv(kamping::send_buf(data));
  auto global_data_original =
      kamping_comm.gatherv(kamping::send_buf(original_data));

  std::sort(global_data_original.begin(), global_data_original.end());
  return global_data_original == global_data;
  // std::is_sorted(global_data.begin(), global_data.end());
}

template <typename T>
auto generate_data(size_t n_local, size_t seed) -> std::vector<T> {
  std::mt19937 eng(seed + kamping::world_rank());
  std::uniform_int_distribution<T> dist(0, std::numeric_limits<T>::max());
  std::vector<T> data(n_local);
  auto gen = [&] { return dist(eng); };
  std::generate(data.begin(), data.end(), gen);
  return data;
}

void log_results(std::string const &json_output_path,
                 std::string const &algorithm, size_t n_local, size_t seed,
                 bool correct) {
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
  mpl::environment::comm_world();  // this perform MPI_init, MPL has no other
                                   // way to do it and calls it implicitly when
                                   // first accessing a communicator
  CLI::App app{"Parallel sorting"};
  std::string algorithm;
  app.add_option("--algorithm", algorithm);
  size_t n_local;
  app.add_option("--n_local", n_local);
  size_t seed = 42;
  app.add_option("--seed", seed);
  size_t iterations = 1;
  app.add_option("--iterations", iterations);
  bool check = false;
  app.add_flag("--check", check);
  bool warmup = false;
  app.add_flag("--warmup", warmup);
  std::string json_output_path = "stdout";
  app.add_option("--json_output_path", json_output_path);
  CLI11_PARSE(app, argc, argv);

  using element_type = uint64_t;

  auto original_data = generate_data<element_type>(n_local, seed);
  size_t local_seed = seed + kamping::world_rank() + kamping::world_size();
  bool correct = false;
  auto do_run = [&](auto &&algo) {
    if (check) {
      kamping::measurements::timer().synchronize_and_start("warmup_time");
      auto data = original_data;
      algo(MPI_COMM_WORLD, data, local_seed);
      kamping::measurements::timer().stop_and_append();
      correct = globally_sorted(MPI_COMM_WORLD, data, original_data);
    } else if (warmup) {
      kamping::measurements::timer().synchronize_and_start("warmup_time");
      auto data = original_data;
      algo(MPI_COMM_WORLD, data, local_seed);
      kamping::measurements::timer().stop_and_append();
    }
    for (size_t iteration = 0; iteration < iterations; iteration++) {
      auto data = original_data;
      kamping::measurements::timer().synchronize_and_start("total_time");
      algo(MPI_COMM_WORLD, data, local_seed);
      kamping::measurements::timer().stop_and_append();
    }
  };
  if (algorithm == "mpi") {
    do_run(mpi::sort<element_type>);
  } else if (algorithm == "mpi_new") {
    do_run(mpi_new::sort<element_type>);
  } else if (algorithm == "kamping") {
    do_run(kamping::sort<element_type>);
  } else if (algorithm == "kamping_verbose") {
    do_run(kamping::sort_verbose<element_type>);
#if defined(KAMPING_EXAMPLES_USE_BOOST)
  } else if (algorithm == "boost") {
    do_run(boost::sort<element_type>);
#endif
  } else if (algorithm == "rwth") {
    do_run(rwth::sort<element_type>);
  } else if (algorithm == "mpl") {
    do_run(mpl::sort<element_type>);
  } else {
    throw std::runtime_error("unsupported algorithm");
  }
  log_results(json_output_path, algorithm, n_local, seed, correct);
  return 0;
}
