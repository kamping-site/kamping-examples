#include <kagen/kagen.h>
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

#include "./mpi_spd_formatters.hpp"

using VertexId = kagen::SInt;

struct Graph {
  std::vector<VertexId> xadj;
  std::vector<VertexId> adjncy;
  std::vector<VertexId> vertex_distribution;
  kamping::Communicator<> const &comm;

  auto first_vertex() const { return vertex_distribution[comm.rank()]; }

  auto last_vertex() const { return vertex_distribution[comm.rank() + 1]; }

  bool is_local(VertexId v) const {
    return v >= first_vertex() && v < last_vertex();
  }

  int home_rank(VertexId v) const {
    auto rank = std::distance(vertex_distribution.begin(),
                              std::upper_bound(vertex_distribution.begin(),
                                               vertex_distribution.end(), v)) -
                1;
    return static_cast<int>(rank);
  }

  auto vertices() const {
    return std::ranges::views::iota(first_vertex(), last_vertex());
  }

  auto global_num_vertices() const { return vertex_distribution.back(); }

  auto neighbors(VertexId v) const {
    auto begin = xadj[v - first_vertex()];
    auto end = xadj[v - first_vertex() + 1];
    std::span span{adjncy};
    span = span.subspan(begin, end - begin);
    return span;
  }

  std::vector<int> get_comm_partners() const {
    std::unordered_set<int> comm_partners_set;
    std::vector<int> comm_partners;
    for (auto v : vertices()) {
      for (auto n : neighbors(v)) {
        comm_partners_set.insert(home_rank(n));
      }
    }
    for (const auto &v : comm_partners_set) {
      comm_partners.emplace_back(v);
    }
    std::sort(comm_partners.begin(), comm_partners.end());
    return comm_partners;
  }
};

class Frontier {
 public:
  void add_vertex(VertexId v, int rank) { _data[rank].push_back(v); }
  virtual std::vector<VertexId> exchange() = 0;
  virtual ~Frontier(){};

 protected:
  std::unordered_map<int, std::vector<VertexId>> _data;
};

class FrontierRegularExchange final : public Frontier {
 public:
  FrontierRegularExchange(const kamping::Communicator<std::vector> &comm)
      : _comm{comm} {}
  std::vector<VertexId> exchange() override {
    SPDLOG_DEBUG("exchanging frontier: frontiers={}", _data);
    std::vector<VertexId> send_buffer;
    std::vector<int> send_counts(_comm.size());
    kamping::measurements::timer().start("copy_buffer");
    for (size_t rank = 0; rank < _comm.size(); rank++) {
      auto it = _data.find(static_cast<int>(rank));
      if (it == _data.end()) {
        send_counts[rank] = 0;
        continue;
      }
      auto &local_data = it->second;
      send_buffer.insert(send_buffer.end(), local_data.begin(),
                         local_data.end());
      send_counts[rank] = static_cast<int>(local_data.size());
      std::vector<VertexId>{}.swap(local_data);
    }
    kamping::measurements::timer().stop_and_add();
    _data.clear();
    kamping::measurements::timer().start("alltoall");
    auto new_frontier = _comm.alltoallv(kamping::send_buf(send_buffer),
                                        kamping::send_counts(send_counts));
    kamping::measurements::timer().stop_and_add();
    return new_frontier;
  }

 private:
  kamping::Communicator<std::vector> const &_comm;
};

class FrontierGridExchange final : public Frontier {
 public:
  FrontierGridExchange(const kamping::Communicator<std::vector> &comm)
      : _comm{comm.mpi_communicator()},
        _grid_comm{_comm.make_grid_communicator()} {}
  std::vector<VertexId> exchange() override {
    SPDLOG_DEBUG("exchanging frontier: frontiers={}", _data);
    std::vector<VertexId> send_buffer;
    std::vector<int> send_counts(_comm.size());
    kamping::measurements::timer().start("copy_buffer");
    for (size_t rank = 0; rank < _comm.size(); rank++) {
      auto it = _data.find(static_cast<int>(rank));
      if (it == _data.end()) {
        send_counts[rank] = 0;
        continue;
      }
      auto &local_data = it->second;
      send_buffer.insert(send_buffer.end(), local_data.begin(),
                         local_data.end());
      send_counts[rank] = static_cast<int>(local_data.size());
      std::vector<VertexId>{}.swap(local_data);
    }
    kamping::measurements::timer().stop_and_add();
    _data.clear();
    kamping::measurements::timer().start("alltoall");
    auto new_frontier = _grid_comm.alltoallv(kamping::send_buf(send_buffer),
                                             kamping::send_counts(send_counts));
    kamping::measurements::timer().stop_and_add();
    return {};
  }

 private:
  kamping::Communicator<std::vector, kamping::plugin::GridCommunicator> _comm;
  kamping::plugin::grid::GridCommunicator<std::vector> _grid_comm;
};

class FrontierSparseExchange final : public Frontier {
 public:
  FrontierSparseExchange(const kamping::Communicator<std::vector> &comm)
      : _comm{comm.mpi_communicator()} {}

  std::vector<VertexId> exchange() override {
    using namespace kamping::plugin::sparse_alltoall;
    SPDLOG_DEBUG("exchanging frontier: frontiers={}", _data);
    kamping::measurements::timer().start("alltoall");
    std::vector<VertexId> new_frontier;
    _comm.alltoallv_sparse(
        sparse_send_buf(_data), on_message([&](auto &probed_message) {
          auto old_size = static_cast<std::vector<VertexId>::difference_type>(
              new_frontier.size());
          new_frontier.resize(new_frontier.size() +
                              probed_message.recv_count());
          kamping::Span<VertexId> message{new_frontier.begin() + old_size,
                                          new_frontier.end()};
          probed_message.recv(kamping::recv_buf(message));
        }));
    kamping::measurements::timer().stop_and_add();
    _data.clear();
    return new_frontier;
  }

 private:
  kamping::Communicator<std::vector, kamping::plugin::SparseAlltoall> const
      _comm;
};

class FrontierMPINeighborhoodExchange final : public Frontier {
 public:
  FrontierMPINeighborhoodExchange(
      kamping::Communicator<std::vector> const &comm,
      const std::vector<int> &comm_partners)
      : _comm(comm), _comm_partners(comm_partners) {
    kamping::measurements::timer().start("create_topology");

    MPI_Dist_graph_create_adjacent(
        _comm.mpi_communicator(), static_cast<int>(_comm_partners.size()),
        comm_partners.data(), MPI_UNWEIGHTED,
        static_cast<int>(_comm_partners.size()), comm_partners.data(),
        MPI_UNWEIGHTED, MPI_INFO_NULL, false, &_graph_comm);

    kamping::measurements::timer().stop_and_add();

    // create global rank to graph communicator rank lookup table
    for (std::size_t i = 0; i < _comm_partners.size(); ++i) {
      _global_rank_to_graph_rank[_comm_partners[i]] = static_cast<int>(i);
    }
  }

  std::vector<VertexId> exchange() override {
    SPDLOG_DEBUG("exchanging frontier: frontiers={}", _data);
    std::vector<VertexId> send_buffer;
    std::vector<int> send_counts(_comm_partners.size(), 0);
    kamping::measurements::timer().start("copy_buffer");
    for (size_t rank = 0; rank < _comm.size(); rank++) {
      auto it = _data.find(static_cast<int>(rank));
      if (it == _data.end()) {
        continue;
      }
      auto &local_data = it->second;
      send_buffer.insert(send_buffer.end(), local_data.begin(),
                         local_data.end());
      send_counts[graph_rank(rank)] = static_cast<int>(local_data.size());
      std::vector<VertexId>{}.swap(local_data);
    }
    _data.clear();
    kamping::measurements::timer().stop_and_add();

    kamping::measurements::timer().start("alltoall");

    // exchange recv counts
    std::vector<int> recv_counts(_comm_partners.size());
    MPI_Neighbor_alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1,
                          MPI_INT, _graph_comm);

    // compute send/recv displacements
    std::vector<int> send_displs(_comm_partners.size(), 0);
    std::vector<int> recv_displs(_comm_partners.size(), 0);
    std::exclusive_scan(send_counts.begin(), send_counts.end(),
                        send_displs.begin(), int(0));
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(),
                        recv_displs.begin(), int(0));

    // perform actual frontier exchange
    const size_t recv_buf_size =
        static_cast<size_t>(recv_displs.back() + recv_counts.back());
    std::vector<VertexId> recv_buf(recv_buf_size);
    MPI_Neighbor_alltoallv(
        send_buffer.data(), send_counts.data(), send_displs.data(),
        kamping::mpi_type_traits<VertexId>::data_type(), recv_buf.data(),
        recv_counts.data(), recv_displs.data(),
        kamping::mpi_type_traits<VertexId>::data_type(), _graph_comm);

    kamping::measurements::timer().stop_and_add();
    return recv_buf;
  }

 private:
  size_t graph_rank(size_t global_rank) {
    auto it = _global_rank_to_graph_rank.find(static_cast<int>(global_rank));
    KASSERT(it != _global_rank_to_graph_rank.end());
    return static_cast<size_t>(it->second);
  }

 private:
  kamping::Communicator<std::vector> const &_comm;
  std::vector<int> _comm_partners;
  std::unordered_map<int, int> _global_rank_to_graph_rank;
  MPI_Comm _graph_comm;
};

class FrontierAlltoallwExchange final : public Frontier {
 public:
  FrontierAlltoallwExchange(const kamping::Communicator<std::vector> &comm)
      : _comm{comm} {}

  std::vector<VertexId> exchange() override {
    SPDLOG_DEBUG("exchanging frontier: frontiers={}", _data);

    std::vector<int> send_counts(_comm.size(), 0);
    std::vector<int> send_displs(_comm.size(), 0);
    std::vector<MPI_Datatype> send_types(_comm.size(), MPI_BYTE);

    std::vector<int> recv_counts(_comm.size(), 0);
    auto &real_send_counts = recv_counts;

    for (size_t rank = 0; rank < _comm.size(); rank++) {
      auto it = _data.find(static_cast<int>(rank));
      if (it == _data.end()) {
        continue;
      }
      auto &local_data = it->second;
      int block_length = static_cast<int>(local_data.size());
      MPI_Aint address;
      MPI_Get_address(local_data.data(), &address);
      kamping::measurements::timer().start("type_creation");
      MPI_Type_create_hindexed(1, &block_length, &address,
                               kamping::mpi_type_traits<VertexId>::data_type(),
                               &send_types[rank]);
      kamping::measurements::timer().stop_and_add();
      kamping::measurements::timer().start("type_commit");
      MPI_Type_commit(&send_types[rank]);
      kamping::measurements::timer().stop_and_add();
      send_counts[rank] = 1;
      real_send_counts[rank] = static_cast<int>(local_data.size());
    }
    kamping::measurements::timer().start("alltoall");
    _comm.alltoall(kamping::send_recv_buf(real_send_counts));
    std::vector<int> recv_displs(_comm.size());
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(),
                        recv_displs.begin(), 0);
    size_t recv_count =
        static_cast<size_t>(recv_displs.back() + recv_counts.back());
    std::vector<VertexId> recv_buf(recv_count);
    std::vector<MPI_Datatype> recv_types(
        _comm.size(), kamping::mpi_type_traits<VertexId>::data_type());
    std::for_each(recv_displs.begin(), recv_displs.end(), [](auto &displ) {
      displ *= static_cast<int>(sizeof(VertexId));
    });
    SPDLOG_DEBUG("send_counts={}, send_displs={}", send_counts, send_displs);
    SPDLOG_DEBUG("recv_counts={}, recv_displs={}", recv_counts, recv_displs);
    MPI_Alltoallw(MPI_BOTTOM, send_counts.data(), send_displs.data(),
                  send_types.data(), recv_buf.data(), recv_counts.data(),
                  recv_displs.data(), recv_types.data(),
                  _comm.mpi_communicator());
    kamping::measurements::timer().stop_and_add();
    _data.clear();
    SPDLOG_DEBUG("recv_buf={}", recv_buf);
    for (auto &type : send_types) {
      if (type != MPI_BYTE) {
        kamping::measurements::timer().start("type_free");
        MPI_Type_free(&type);
        kamping::measurements::timer().stop_and_add();
      }
    }
    return recv_buf;
  }

 private:
  kamping::Communicator<std::vector> const &_comm;
};

enum class ExchangeType { NoCopy, Sparse, Regular, MPINeighborhood, Grid };

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
              {"no_copy", ExchangeType::NoCopy},
              {"sparse", ExchangeType::Sparse},
              {"neighborhood", ExchangeType::MPINeighborhood},
              {"grid", ExchangeType::Grid},
              {"regular", ExchangeType::Regular}}));
  size_t iterations = 1;
  app.add_option("--iterations", iterations, "Number of iterations");
  std::string json_output_path = "stdout";
  app.add_option("--json_output_path", json_output_path, "Path to JSON output");

  CLI11_PARSE(app, argc, argv);
  kagen::KaGen kagen(MPI_COMM_WORLD);
  kagen.UseCSRRepresentation();
  auto graph = kagen.GenerateFromOptionString(kagen_option_string);
  std::vector xadj = graph.TakeXadj<VertexId>();
  std::vector adjncy = graph.TakeAdjncy<VertexId>();
  auto dist = kagen::BuildVertexDistribution<VertexId>(
      graph, kamping::mpi_type_traits<VertexId>::data_type(), MPI_COMM_WORLD);
  Graph g{std::move(xadj), std::move(adjncy), std::move(dist),
          kamping::comm_world()};
  kamping::Communicator<std::vector, kamping::plugin::GridCommunicator,
                        kamping::plugin::SparseAlltoall>
      comm;
  std::default_random_engine gen(seed);
  std::uniform_int_distribution<VertexId> vertex_dist(0,
                                                      g.global_num_vertices());
  VertexId root = vertex_dist(gen);
  SPDLOG_DEBUG("root={}", root);
  SPDLOG_DEBUG("[{}, {})", g.first_vertex(), g.last_vertex());
  SPDLOG_DEBUG("vertices={}, ranks={}", g.vertices(),
               g.vertices() | std::views::transform(
                                  [&](auto v) { return g.home_rank(v); }));
  constexpr size_t unreachable = std::numeric_limits<size_t>::max();
  for (size_t iteration = 0; iteration < iterations; iteration++) {
    kamping::measurements::timer().synchronize_and_start("bfs");
    std::vector<VertexId> current_frontier;
    if (g.is_local(root)) {
      current_frontier.push_back(root);
    }
    std::vector<size_t> bfs_level(g.last_vertex() - g.first_vertex(),
                                  unreachable);

    auto frontier = [&]() -> std::unique_ptr<Frontier> {
      switch (exchange_type) {
        case ExchangeType::MPINeighborhood:
          return std::make_unique<FrontierMPINeighborhoodExchange>(
              FrontierMPINeighborhoodExchange{kamping::comm_world(),
                                              g.get_comm_partners()});
        case ExchangeType::Regular:
          return std::make_unique<FrontierRegularExchange>(
              FrontierRegularExchange{kamping::comm_world()});
        case ExchangeType::NoCopy:
          return std::make_unique<FrontierAlltoallwExchange>(
              FrontierAlltoallwExchange{kamping::comm_world()});
        case ExchangeType::Sparse:
          return std::make_unique<FrontierSparseExchange>(
              FrontierSparseExchange{kamping::comm_world()});
        case ExchangeType::Grid: {
          kamping::measurements::timer().start("create_grid");
          auto frontier_ = std::make_unique<FrontierGridExchange>(
              FrontierGridExchange{kamping::comm_world()});
          kamping::measurements::timer().stop_and_append();
          return frontier_;
        }
        default:
          KASSERT(false, "should never be called");
          return std::make_unique<FrontierRegularExchange>(
              FrontierRegularExchange{kamping::comm_world()});
      }
    }();
    size_t level = 0;
    while (!comm.allreduce_single(kamping::send_buf(current_frontier.empty()),
                                  kamping::op(std::logical_and<>{}))) {
      SPDLOG_DEBUG("frontier={}", current_frontier);
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
      current_frontier = frontier->exchange();
      level++;
      SPDLOG_DEBUG("level={}", level);
    }
    kamping::measurements::timer().stop_and_append();
  }

  std::unique_ptr<std::ostream> output_stream;
  if (json_output_path == "stdout") {
    output_stream = std::make_unique<std::ostream>(std::cout.rdbuf());
  } else {
    std::ofstream file_output(json_output_path);
    output_stream = std::make_unique<std::ofstream>(std::move(file_output));
  }
  if (kamping::comm_world().is_root()) {
    *output_stream << "{\n";
  }
  kamping::measurements::timer().aggregate_and_print(
      kamping::measurements::SimpleJsonPrinter<>{*output_stream});
  if (kamping::comm_world().is_root()) {
    *output_stream << ",\n";
    *output_stream << "\"info\": {\n";
    *output_stream << "  \"exchange_type\": ";
    switch (exchange_type) {
      case ExchangeType::NoCopy:
        *output_stream << "\"no_copy\"";
        break;
      case ExchangeType::Sparse:
        *output_stream << "\"sparse\"";
        break;
      case ExchangeType::Regular:
        *output_stream << "\"regular\"";
        break;
      case ExchangeType::MPINeighborhood:
        *output_stream << "\"neighborhood\"";
        break;
      case ExchangeType::Grid:
        *output_stream << "\"grid\"";
        break;
    }
    *output_stream << ",\n";
    *output_stream << "  \"p\": " << kamping::comm_world().size() << ",\n";
    *output_stream << "  \"kagen_option_string\": \"" << kagen_option_string
                   << "\",\n";
    *output_stream << "  \"seed\": " << seed << "\n";
    *output_stream << "}\n";
    *output_stream << "}";
  }
  // if (comm.is_root()) {
  //   std::cout << "\n";
  // }
  // auto max_bfs_level = comm.reduce_single(kamping::send_buf(level),
  //                                         kamping::op(kamping::ops::max<>{}));
  // if (max_bfs_level) {
  //   std::cout << "max_bfs_level=" << max_bfs_level.value() << std::endl;
  // }
  return 0;
}
