#include <CLI/CLI.hpp>
#include <kagen/kagen.h>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/reduce.hpp>
#include <kamping/communicator.hpp>
#include <kamping/environment.hpp>
#include <kamping/mpi_datatype.hpp>
#include <ranges>
#include <spdlog/fmt/ranges.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/spdlog.h>

class rank_formatter : public spdlog::custom_flag_formatter {
public:
  void format(const spdlog::details::log_msg &, const std::tm &,
              spdlog::memory_buf_t &dest) override {
    auto rank_string = std::to_string(kamping::comm_world().rank());
    dest.append(rank_string);
  }
  std::unique_ptr<custom_flag_formatter> clone() const override {
    return spdlog::details::make_unique<rank_formatter>();
  }
};
class size_formatter : public spdlog::custom_flag_formatter {
public:
  void format(const spdlog::details::log_msg &, const std::tm &,
              spdlog::memory_buf_t &dest) override {
    auto size_string = std::to_string(kamping::comm_world().size());
    dest.append(size_string);
  }
  std::unique_ptr<custom_flag_formatter> clone() const override {
    return spdlog::details::make_unique<size_formatter>();
  }
};

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
    return std::distance(vertex_distribution.begin(),
                         std::upper_bound(vertex_distribution.begin(),
                                          vertex_distribution.end(), v)) -
           1;
  }

  auto vertices() const {
    return std::ranges::views::iota(first_vertex(), last_vertex());
  }

  auto neighbors(VertexId v) const {
    auto begin = xadj[v - first_vertex()];
    auto end = xadj[v - first_vertex() + 1];
    std::span span{adjncy};
    span = span.subspan(begin, end - begin);
    return span;
  }
};

struct Frontier {
  std::unordered_map<int, std::vector<VertexId>> data;
  kamping::Communicator<> const &comm;

  Frontier(kamping::Communicator<> const &comm) : comm(comm) {}
  void add_vertex(VertexId v, int rank) { data[rank].push_back(v); }

  std::vector<VertexId> exchange() {
    spdlog::debug("exchanging frontier: frontiers={}", data);
    std::vector<VertexId> send_buffer;
    std::vector<int> send_counts(comm.size());
    for (size_t rank = 0; rank < comm.size(); rank++) {
      auto it = data.find(rank);
      if (it == data.end()) {
        send_counts[rank] = 0;
        continue;
      }
      auto &local_data = it->second;
      send_buffer.insert(send_buffer.end(), local_data.begin(),
                         local_data.end());
      send_counts[rank] = local_data.size();
      std::vector<VertexId>{}.swap(local_data);
    }
    data.clear();
    auto new_frontier = comm.alltoallv(kamping::send_buf(send_buffer),
                                       kamping::send_counts(send_counts));
    return new_frontier;
  }

  std::vector<VertexId> exchange_no_copy() {
    spdlog::debug("exchanging frontier: frontiers={}", data);

    std::vector<int> send_counts(comm.size());
    std::vector<int> send_displs(comm.size());
    std::vector<MPI_Aint> addresses(comm.size());
    for (size_t rank = 0; rank < comm.size(); rank++) {
      auto it = data.find(rank);
      if (it == data.end()) {
        send_counts[rank] = 0;
        continue;
      }
      MPI_Aint addr;
      MPI_Get_address(it->second.data(), &addr);
      addresses[rank] = addr;
      send_counts[rank] = it->second.size();
    }
    spdlog::debug("addresses={}", addresses);
    auto non_zero_addresses =
        addresses | std::views::filter([](auto addr) { return addr != 0; });
    for (size_t rank = 0; rank < comm.size(); rank++) {
      if (send_counts[rank] == 0) {
        continue;
      }
      auto diff = addresses[rank] - min_address;
      spdlog::debug("rank={}, diff={}", rank, diff);
      send_displs[rank] = kamping::asserting_cast<int>(diff);
    }
    std::vector<int> recv_counts =
        comm.alltoall(kamping::send_buf(send_counts));
    std::vector<int> recv_displs(comm.size());
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(),
                        recv_displs.begin(), 0);
    size_t recv_count = recv_displs.back() + recv_counts.back();
    std::vector<VertexId> new_frontier(recv_count);
    MPI_Alltoallv_c(MP_BOTTOM, send_counts.data(), send_displs.data(),
                  kamping::mpi_type_traits<VertexId>::data_type(),
                  new_frontier.data(), recv_counts.data(), recv_displs.data(),
                  kamping::mpi_type_traits<VertexId>::data_type(),
                  comm.mpi_communicator());
    data.clear();
    return new_frontier;
  }
  std::vector<VertexId> exchange_no_copy_datatype() {
    std::vector<int> send_counts(comm.size(), 0);
    std::vector<int> send_displs(comm.size(), 0);
    std::vector<MPI_Datatype> send_types(comm.size(), MPI_INT);

    std::vector<int> recv_counts(comm.size(), 0);
    auto &real_send_counts = recv_counts;

    for (size_t rank = 0; rank < comm.size(); rank++) {
      auto it = data.find(rank);
      if (it == data.end()) {
        continue;
      }
      auto &local_data = it->second;
      int block_length = static_cast<int>(local_data.size());
      MPI_Aint address;
      MPI_Get_address(local_data.data(), &address);
      MPI_Type_create_hindexed(1, &block_length, &address,
                               kamping::mpi_type_traits<VertexId>::data_type(),
                               &send_types[rank]);
      MPI_Type_commit(&send_types[rank]);
      send_counts[rank] = 1;
      real_send_counts[rank] = local_data.size();
    }
    comm.alltoall(kamping::send_recv_buf(real_send_counts));
    std::vector<int> recv_displs(comm.size());
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(),
                        recv_displs.begin(), 0);
    auto recv_count = recv_displs.back() + recv_counts.back();
    std::vector<VertexId> recv_buf(recv_count);
    std::vector<MPI_Datatype> recv_types(
        comm.size(), kamping::mpi_type_traits<VertexId>::data_type());
    MPI_Alltoallw(MPI_BOTTOM, send_counts.data(), send_displs.data(),
                  send_types.data(), recv_buf.data(), recv_counts.data(),
                  recv_displs.data(), recv_types.data(),
                  comm.mpi_communicator());
    for (auto& type : send_types) {
      if (type != MPI_INT) {
        MPI_Type_free(&type);
      }
    }
    return recv_buf;
  }
};

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
  auto const &comm = kamping::comm_world();
  VertexId root = 0;
  std::vector<VertexId> current_frontier;
  spdlog::debug("[{}, {})", g.first_vertex(), g.last_vertex());
  spdlog::debug("vertices={}, ranks={}", g.vertices(),
                g.vertices() | std::views::transform(
                                   [&](auto v) { return g.home_rank(v); }));
  if (g.is_local(root)) {
    current_frontier.push_back(root);
  }
  constexpr size_t unreachable = std::numeric_limits<size_t>::max();
  std::vector<size_t> bfs_level(g.last_vertex() - g.first_vertex(),
                                unreachable);

  Frontier frontier{kamping::comm_world()};
  size_t level = 0;
  while (!comm.allreduce_single(kamping::send_buf(current_frontier.empty()),
                                kamping::op(std::logical_and<>{}))) {
    spdlog::debug("frontier={}", current_frontier);
    for (auto v : current_frontier) {
      spdlog::debug("visiting {}", v);
      KASSERT(g.is_local(v));
      if (bfs_level[v - g.first_vertex()] != unreachable) {
        continue;
      }
      bfs_level[v - g.first_vertex()] = level;
      for (auto u : g.neighbors(v)) {
        int rank = g.home_rank(u);
        spdlog::debug("adding {} to frontier of {}", u, rank);
        frontier.add_vertex(u, rank);
      }
    }
    current_frontier = frontier.exchange_no_copy_datatype();
    level++;
    spdlog::debug("level={}", level);
  }
  auto max_bfs_level = comm.reduce_single(kamping::send_buf(level),
                                          kamping::op(kamping::ops::max<>{}));
  if (max_bfs_level) {
    std::cout << "max_bfs_level=" << max_bfs_level.value() << std::endl;
  }
  return 0;
}
