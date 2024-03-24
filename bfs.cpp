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

#include "bfs/bfs_algorithm.hpp"
#include "bfs/common.hpp"
#include "bfs/kamping.hpp"
#include "bfs/kamping_flattened.hpp"
#include "bfs/kamping_grid.hpp"
#include "bfs/kamping_sparse.hpp"
#include "bfs/mpi.hpp"
#include "bfs/mpl.hpp"
#include "bfs/rwth_mpi.hpp"
#include "bfs/utils.hpp"

enum class Algorithm {
  boost,
  kamping,
  kamping_flattened,
  kamping_grid,
  kamping_sparse,
  mpi,
  mpi_neighborhood,
  mpl,
  rwth_mpi
};

std::string to_string(const Algorithm& algorithm) {
  switch (algorithm) {
    case Algorithm::boost:
      return "boost";
    case Algorithm::kamping:
      return "kamping";
    case Algorithm::kamping_flattened:
      return "kamping_flattened";
    case Algorithm::kamping_grid:
      return "kamping_grid";
    case Algorithm::kamping_sparse:
      return "kamping_sparse";
    case Algorithm::mpi:
      return "mpi";
    case Algorithm::mpi_neighborhood:
      return "mpi_neighborhood";
    case Algorithm::mpl:
      return "mpl";
    case Algorithm::rwth_mpi:
      return "rwth_mpi";
    default:
      throw std::runtime_error("unsupported algorithm");
  };
}

auto print_on_root = [](const std::string& msg) {
  kamping::comm_world().barrier();
  if (kamping::comm_world().is_root()) {
    std::cout << msg << std::endl;
  }
};

auto dispatch_bfs_algorithm(Algorithm algorithm) {
  using namespace graph;
  switch (algorithm) {
#if defined(KAMPING_EXAMPLES_USE_BOOST)
    case Algorithm::boost: {
      using Frontier = bfs_boost::BFSFrontier;
      return bfs::bfs<Frontier>;
    }
#endif
    case Algorithm::kamping: {
      using Frontier = bfs_kamping::BFSFrontier;
      return bfs::bfs<Frontier>;
    }
    case Algorithm::kamping_flattened: {
      using Frontier = bfs_kamping_flattened::BFSFrontier;
      return bfs::bfs<Frontier>;
    }
    case Algorithm::kamping_grid: {
      using Frontier = bfs_kamping_grid::BFSFrontier;
      return bfs::bfs<Frontier>;
    }
    case Algorithm::kamping_sparse: {
      using Frontier = bfs_kamping_sparse::BFSFrontier;
      return bfs::bfs<Frontier>;
    }
    case Algorithm::mpi: {
      using Frontier = bfs_mpi::BFSFrontier;
      return bfs::bfs<Frontier>;
    }
    case Algorithm::mpi_neighborhood: {
      using Frontier = bfs_mpi_neighborhood::BFSFrontier;
      return bfs::bfs<Frontier>;
    }
    case Algorithm::mpl: {
      using Frontier = bfs_mpl::BFSFrontier;
      return bfs::bfs<Frontier>;
    }
    case Algorithm::rwth_mpi: {
      using Frontier = bfs_rwth_mpi::BFSFrontier;
      return bfs::bfs<Frontier>;
    }
    default:
      throw std::runtime_error("unsupported algorithm");
  };
}

void log_results(std::string const& json_output_path, std::size_t iterations,
                 std::string const& algorithm,
                 std::string const& kagen_option_string, size_t max_bfs_level,
                 size_t seed) {
  std::unique_ptr<std::ostream> output_stream;
  print_on_root("\nstart messing with output");
  if (json_output_path == "stdout") {
    output_stream = std::make_unique<std::ostream>(std::cout.rdbuf());
  } else {
    std::ofstream file_output(json_output_path);
    output_stream = std::make_unique<std::ofstream>(std::move(file_output));
  }
  print_on_root("\nstop messing with output");
  if (kamping::comm_world().rank() == 0) {
    *output_stream << "{\n";
  }
  print_on_root("\nstart timer gathering");
  kamping::measurements::timer().aggregate_and_print(
      kamping::measurements::SimpleJsonPrinter<>{*output_stream});
  print_on_root("\nfinished timer gathering");
  if (mpl::environment::comm_world().rank() == 0) {
    *output_stream << ",\n";
    *output_stream << "\"info\": {\n";
    *output_stream << "  \"iterations\": "
                   << "\"" << iterations << "\",\n";
    *output_stream << "  \"algorithm\": "
                   << "\"" << algorithm << "\",\n";
    *output_stream << "  \"graph\": "
                   << "\"" << kagen_option_string << "\",\n";
    *output_stream << "  \"p\": " << mpl::environment::comm_world().size()
                   << ",\n";
    *output_stream << "  \"max_bfs_level\": " << max_bfs_level << ",\n";
    *output_stream << "  \"seed\": " << seed << "\n";
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
  bool permute = false;
  app.add_flag("--permute", permute);
  size_t seed = 42;
  app.add_option("--seed", seed);
  Algorithm algorithm = Algorithm::mpi;
  app.add_option("--algorithm", algorithm, "Algorithm type")
      ->transform(
          CLI::CheckedTransformer(std::unordered_map<std::string, Algorithm>{
              {"boost", Algorithm::boost},
              {"kamping", Algorithm::kamping},
              {"kamping_flattened", Algorithm::kamping_flattened},
              {"kamping_grid", Algorithm::kamping_grid},
              {"kamping_sparse", Algorithm::kamping_sparse},
              {"mpi", Algorithm::mpi},
              {"mpi_neighborhood", Algorithm::mpi_neighborhood},
              {"mpl", Algorithm::mpl},
              {"rwth_mpi", Algorithm::rwth_mpi}}));
  size_t iterations = 1;
  app.add_option("--iterations", iterations, "Number of iterations");
  std::string json_output_path = "stdout";
  app.add_option("--json_output_path", json_output_path, "Path to JSON output");
  CLI11_PARSE(app, argc, argv);

  auto do_run = [&](auto&& bfs) {
    print_on_root("start graph gen");
    const auto g = [&]() {
      auto graph = graph::generate_distributed_graph(kagen_option_string);
      if (permute) {
        kagen_option_string += ";permute=true";
        return graph::permute(graph, seed);
      } else {
        kagen_option_string += ";permute=false";
        return graph;
      }
    }();

    const graph::VertexId root = [&]() {
      graph::VertexId r = graph::generate_start_vertex(g, seed);
      if (permute) {
        return graph::permute_vertex(g.global_num_vertices(), seed, r);
      } else {
        return r;
      }
    }();

    std::vector<size_t> bfs_levels;
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
      kamping::measurements::timer().synchronize_and_start("total_time");
      bfs_levels = bfs(g, root, MPI_COMM_WORLD);
      kamping::measurements::timer().stop_and_append();
    }
    print_on_root("finished run");
    const size_t max_num_comm_partners = kamping::comm_world().allreduce_single(
        kamping::send_buf(g.get_comm_partners().size()),
        kamping::op(kamping::ops::max<>{}));
    print_on_root("max num comm partners: " +
                  std::to_string(max_num_comm_partners));
    return bfs_levels;
  };

  auto bfs_levels = do_run(dispatch_bfs_algorithm(algorithm));

  print_on_root("start max_level computation");
  // outputting
  auto reached_levels = bfs_levels | std::views::filter([](auto l) noexcept {
                          return l != graph::unreachable_vertex;
                        });
  auto it = std::ranges::max_element(reached_levels);
  size_t max_bfs_level = it == reached_levels.end() ? 0 : *it;
  print_on_root("mid max_level computation");
  kamping::comm_world().allreduce(kamping::send_recv_buf(max_bfs_level),
                                  kamping::op(kamping::ops::max<>{}));
  print_on_root("finished max_level computation");
  log_results(json_output_path, iterations, to_string(algorithm),
              kagen_option_string, max_bfs_level, seed);
  return 0;
}
