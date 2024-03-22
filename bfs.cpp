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
  MPI,
  kamping,
  kamping_flattened,
  kamping_sparse,
  kamping_grid,
  rwth_mpi,
  mpl
};

std::vector<size_t> dispatch_bfs_algorithm(
    Algorithm algorithm, const std::string& kagen_option_string, size_t seed) {
  using namespace graph;

  auto g = graph::generate_distributed_graph(kagen_option_string);
  const graph::VertexId root = graph::generate_start_vertex(g, seed);

  switch (algorithm) {
    case Algorithm::MPI: {
      using Frontier = bfs_mpi::BFSFrontier;
      return bfs<Frontier>(g, root, MPI_COMM_WORLD);
    }
    case Algorithm::kamping: {
      using Frontier = bfs_kamping::BFSFrontier;
      return bfs<Frontier>(g, root, MPI_COMM_WORLD);
    }
    case Algorithm::kamping_flattened: {
      using Frontier = bfs_kamping_flattened::BFSFrontier;
      return bfs<Frontier>(g, root, MPI_COMM_WORLD);
    }
    case Algorithm::kamping_sparse: {
      using Frontier = bfs_kamping::BFSFrontier;
      return bfs<Frontier>(g, root, MPI_COMM_WORLD);
    }
    case Algorithm::kamping_grid: {
      using Frontier = bfs_kamping::BFSFrontier;
      return bfs<Frontier>(g, root, MPI_COMM_WORLD);
    }
    case Algorithm::rwth_mpi: {
      using Frontier = bfs_rwth_mpi::BFSFrontier;
      return bfs<Frontier>(g, root, MPI_COMM_WORLD);
    }
    case Algorithm::mpl: {
      using Frontier = bfs_mpl::BFSFrontier;
      return bfs<Frontier>(g, root, MPI_COMM_WORLD);
    }
    default:
      std::abort();
  };
}

auto main(int argc, char* argv[]) -> int {
  auto formatter = std::make_unique<spdlog::pattern_formatter>();
  formatter->add_flag<rank_formatter>('r');
  formatter->add_flag<size_formatter>('s');
  formatter->set_pattern("[%r/%s] [%^%l%$] %v");
  spdlog::set_formatter(std::move(formatter));

  spdlog::default_logger()->set_level(spdlog::level::debug);
  kamping::Environment env;
  CLI::App app{"BFS"};
  std::string kagen_option_string;
  app.add_option("--kagen_option_string", kagen_option_string, "Kagen options")
      ->required();
  size_t seed = 42;
  app.add_option("--seed", seed);
  Algorithm algorithm = Algorithm::MPI;
  app.add_option("--algorithm", algorithm, "Algorithm type")
      ->transform(
          CLI::CheckedTransformer(std::unordered_map<std::string, Algorithm>{
              {"mpi", Algorithm::MPI},
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

  auto bfs_levels = dispatch_bfs_algorithm(algorithm, kagen_option_string, seed);

  // outputting
  auto reached_levels = bfs_levels | std::views::filter([](auto l) noexcept {
                          return l != graph::unreachable_vertex;
                        });
  auto it = std::ranges::max_element(reached_levels);
  size_t max_bfs_level = it == reached_levels.end() ? 0 : *it;
  kamping::comm_world().allreduce(kamping::send_recv_buf(max_bfs_level),
                                  kamping::op(kamping::ops::max<>{}));
  if (kamping::comm_world().is_root()) {
    std::cout << "\n";
    std::cout << "max_bfs_level=" << max_bfs_level << std::endl;
  }
  return 0;
}
