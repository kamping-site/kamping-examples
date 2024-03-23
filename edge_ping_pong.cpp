#if defined(KAMPING_EXAMPLES_USE_BOOST)
#include "bfs/boost.hpp"
#endif

#include <spdlog/fmt/ranges.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/reduce.hpp>
#include <kamping/communicator.hpp>
#include <kamping/environment.hpp>
#include <kamping/measurements/printer.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/mpi_datatype.hpp>
#include <kamping/plugin/alltoall_grid.hpp>
#include <kamping/plugin/alltoall_sparse.hpp>
#include <memory>
#include <ranges>

#include "bfs/common.hpp"
#include "bfs/kamping.hpp"
#include "bfs/kamping_flattened.hpp"
#include "bfs/kamping_grid.hpp"
#include "bfs/mpi.hpp"
#include "bfs/mpl.hpp"
#include "bfs/rwth_mpi.hpp"

enum class Algorithm {
  mpi,
  kamping,
  kamping_flattened,
  kamping_sparse,
  kamping_grid,
  rwth_mpi,
  mpl,
  boost
};

std::string to_string(const Algorithm& algorithm) {
  switch (algorithm) {
    case Algorithm::mpi:
      return "mpi";
    case Algorithm::kamping:
      return "kamping";
    case Algorithm::kamping_flattened:
      return "kamping_flattened";
    case Algorithm::kamping_sparse:
      return "kamping_sparse";
    case Algorithm::kamping_grid:
      return "kamping_grid";
    case Algorithm::rwth_mpi:
      return "rwth_mpi";
    case Algorithm::mpl:
      return "mpl";
    case Algorithm::boost:
      return "boost";
    default:
      throw std::runtime_error("unsupported algorithm");
  };
}

auto dispatch_bfs_algorithm(Algorithm algorithm) {
  using namespace graph;
  switch (algorithm) {
    case Algorithm::mpi: {
      using Frontier = bfs_mpi::BFSFrontier;
      return graph_ping_pong<Frontier>;
    }
    case Algorithm::kamping: {
      using Frontier = bfs_kamping::BFSFrontier;
      return graph_ping_pong<Frontier>;
    }
    case Algorithm::kamping_flattened: {
      using Frontier = bfs_kamping_flattened::BFSFrontier;
      return graph_ping_pong<Frontier>;
    }
    case Algorithm::kamping_sparse: {
      using Frontier = bfs_kamping::BFSFrontier;
      return graph_ping_pong<Frontier>;
    }
    case Algorithm::kamping_grid: {
      using Frontier = bfs_kamping::BFSFrontier;
      return graph_ping_pong<Frontier>;
    }
    case Algorithm::rwth_mpi: {
      using Frontier = bfs_rwth_mpi::BFSFrontier;
      return graph_ping_pong<Frontier>;
    }
    case Algorithm::mpl: {
      using Frontier = bfs_mpl::BFSFrontier;
      return graph_ping_pong<Frontier>;
    }
#if defined(KAMPING_EXAMPLES_USE_BOOST)
    case Algorithm::boost: {
      using Frontier = bfs_boost::BFSFrontier;
      return graph_ping_pong<Frontier>;
    }
#endif
    default:
      throw std::runtime_error("unsupported algorithm");
  };
}

void log_results(std::string const& json_output_path,
                 std::string const& algorithm, size_t seed) {
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
    *output_stream << "  \"seed\": " << seed << ",\n";
    *output_stream << "}\n";
    *output_stream << "}";
  }
}

auto main(int argc, char* argv[]) -> int {
  mpl::environment::comm_world();  // this perform MPI_init, MPL has no other
                                   // way to do it and calls it implicitly when
                                   // first accessing a communicator

  auto formatter = std::make_unique<spdlog::pattern_formatter>();
  formatter->add_flag<rank_formatter>('r');
  formatter->add_flag<size_formatter>('s');
  formatter->set_pattern("[%r/%s] [%^%l%$] %v");
  spdlog::set_formatter(std::move(formatter));

  spdlog::default_logger()->set_level(spdlog::level::debug);
  CLI::App app{"BFS"};
  std::string kagen_option_string;
  app.add_option("--kagen_option_string", kagen_option_string, "Kagen options")
      ->required();
  size_t seed = 42;
  app.add_option("--seed", seed);
  Algorithm algorithm = Algorithm::mpi;
  app.add_option("--algorithm", algorithm, "Algorithm type")
      ->transform(
          CLI::CheckedTransformer(std::unordered_map<std::string, Algorithm>{
              {"boost", Algorithm::boost},
              {"mpi", Algorithm::mpi},
              {"kamping", Algorithm::kamping},
              {"kamping_flattened", Algorithm::kamping_flattened},
              {"kamping_sparse", Algorithm::kamping_sparse},
              {"kamping_grid", Algorithm::kamping_sparse},
              {"rwth_mpi", Algorithm::rwth_mpi},
              {"mpl", Algorithm::mpl}}));
  size_t iterations = 1;
  app.add_option("--iterations", iterations, "Number of iterations");
  std::string json_output_path = "stdout";
  app.add_option("--json_output_path", json_output_path, "Path to JSON output");
  CLI11_PARSE(app, argc, argv);

  auto do_run = [&](auto&& edge_ping_pong) {
    const auto g = graph::generate_distributed_graph(kagen_option_string);
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
      kamping::measurements::timer().synchronize_and_start("total_time");
      edge_ping_pong(g, MPI_COMM_WORLD);
      kamping::measurements::timer().stop_and_append();
    }
  };

  do_run(dispatch_bfs_algorithm(algorithm));
  log_results(json_output_path, to_string(algorithm), seed);
  return 0;
}
