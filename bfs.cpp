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
#include <random>
#include <ranges>

#include "bfs/common.hpp"
#include "bfs/mpi.hpp"

enum class ExchangeType { MPI, NoCopy, Sparse, Regular, MPINeighborhood, Grid };

template <typename Frontier, typename Comm>
auto execute_bfs(Frontier &frontier, const Comm &comm, const graph::Graph &g,
                 graph::VertexId root) {
  using namespace graph;

  constexpr size_t unreachable = std::numeric_limits<size_t>::max();
  kamping::measurements::timer().synchronize_and_start("bfs");
  std::vector<VertexId> current_frontier;
  if (g.is_local(root)) {
    current_frontier.push_back(root);
  }
  std::vector<size_t> bfs_level(g.last_vertex() - g.first_vertex(),
                                unreachable);
  size_t level = 0;
  while (!comm.allreduce_single(kamping::send_buf(current_frontier.empty()),
                                kamping::op(std::logical_and<>{}))) {
    SPDLOG_DEBUG("frontier={}", current_frontier);

    kamping::measurements::timer().start("local_frontier_processing");
    for (auto v : current_frontier) {
      SPDLOG_DEBUG("visiting {}", v);
      KASSERT(g.is_local(v));
      if (bfs_level[v - g.first_vertex()] != unreachable) {
        continue;
      }
      bfs_level[v - g.first_vertex()] = level;
      for (auto u : g.neighbors(v)) {
        int rank = g.home_rank(u);
        SPDLOG_DEBUG("adding {} to frontier of {}", u, rank);
        frontier->add_vertex(u, rank);
      }
    }
    kamping::measurements::timer().stop_and_append();
    current_frontier = frontier->exchange();
    level++;
    SPDLOG_DEBUG("level={}", level);
  }
  kamping::measurements::timer().stop_and_append();
  return bfs_level;
}

auto main(int argc, char *argv[]) -> int {
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
  ExchangeType exchange_type = ExchangeType::Regular;
  app.add_option("--exchange_type", exchange_type, "Exchange type")
      ->transform(
          CLI::CheckedTransformer(std::unordered_map<std::string, ExchangeType>{
              {"mpi", ExchangeType::MPI},
              {"no_copy", ExchangeType::NoCopy},
              {"sparse", ExchangeType::Sparse},
              {"neighborhood", ExchangeType::MPINeighborhood},
              {"grid", ExchangeType::Grid},
              {"regular", ExchangeType::Regular}}));
  size_t iterations = 1;
  app.add_option("--iterations", iterations, "Number of iterations");
  std::string json_output_path = "stdout";
  app.add_option("--json_output_path", json_output_path, "Path to JSON output");
  //
  CLI11_PARSE(app, argc, argv);

  auto g = graph::generate_distributed_graph(kagen_option_string);
  const graph::VertexId root = graph::generate_start_vertex(g, seed);

  std::vector<size_t> bfs_levels;

  switch (exchange_type) {
    case ExchangeType::MPI:
      bfs_levels = mpi::bfs(g, root, MPI_COMM_WORLD);
      break;
    default:
      std::abort();
  };
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
