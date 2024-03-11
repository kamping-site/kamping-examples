#include <mpi.h>

#include <CLI/CLI.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>
#include <cstddef>
#include <kamping/communicator.hpp>
#include <kamping/mpi_datatype.hpp>
#include <kamping/p2p/recv.hpp>
#include <kamping/p2p/send.hpp>
#include <utility>
#include <vector>

enum class Type {
  create_struct,
  pair_as_bytes,
  contiguous_type,
  serialization,
  builtin
};

std::string to_string(Type t) {
  switch (t) {
    case Type::create_struct:
      return "create_struct";
    case Type::pair_as_bytes:
      return "pair_as_bytes";
    case Type::contiguous_type:
      return "contiguous_type";
    case Type::serialization:
      return "serialization";
    case Type::builtin:
      return "builtin";
  }
  return "unknown";
}

void log_result(std::vector<double>& times, std::size_t n_reps,
                std::size_t data_size, Type mpi_type_constructor,
                std::string const& value_type,
                std::ostream& out) {
  if (kamping::comm_world().is_root()) {
    MPI_Reduce(MPI_IN_PLACE, times.data(), static_cast<int>(n_reps), MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(times.data(), nullptr, static_cast<int>(n_reps), MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
  }
  if (kamping::comm_world().is_root()) {
    double average_time = std::accumulate(times.begin(), times.end(), 0.0) /
                          static_cast<double>(n_reps);
    auto [min_time, max_time] = std::minmax_element(times.begin(), times.end());
    out << "RESULT ";
    out << "n_reps=" << n_reps << " data_size=" << data_size
         << " mpi_type_constructor=" << to_string(mpi_type_constructor)
         << " value_type=" << value_type << " ";
    out << "average_time=" << average_time << " min_time=" << *min_time
         << " max_time=" << *max_time << "\n";
  }
}

template <typename value_type>
auto benchmark(kamping::Communicator<> const& comm, std::size_t n_reps,
               std::size_t data_size, Type mpi_type_constructor)
    -> std::vector<double> {
  std::vector<value_type> data(data_size);
  MPI_Datatype type;
  int num_elements;

  switch (mpi_type_constructor) {
    case Type::create_struct: {
      type = kamping::struct_type<value_type>::data_type();
      kamping::mpi_env.commit_and_register(type);
      num_elements = static_cast<int>(data.size());
      break;
    }
    case Type::pair_as_bytes: {
      type = kamping::byte_serialized<value_type>::data_type();
      kamping::mpi_env.commit_and_register(type);
      num_elements = static_cast<int>(data.size());
      break;
    }
    case Type::contiguous_type:
      MPI_Type_contiguous(
          static_cast<int>(data_size) * static_cast<int>(sizeof(value_type)),
          MPI_BYTE, &type);
      kamping::mpi_env.commit_and_register(type);
      num_elements = 1;
      break;
    case Type::builtin:
      type = MPI_BYTE;
      num_elements = static_cast<int>(data.size() * sizeof(value_type));
      break;
    case Type::serialization:
      break;
  }
  std::vector<double> times(n_reps);
  for (std::size_t i = 0; i < n_reps; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if (comm.is_root()) {
      if (mpi_type_constructor != Type::serialization) {
        comm.send(kamping::send_buf(data), kamping::send_type(type),
                  kamping::send_count(num_elements), kamping::destination(1));
        comm.recv(kamping::recv_buf(data), kamping::recv_type(type),
                  kamping::recv_count(num_elements), kamping::source(1),
                  kamping::tag(comm.default_tag()));
      } else {
        comm.send(kamping::send_buf(kamping::as_serialized(data)),
                  kamping::destination(1));
        comm.recv(kamping::recv_buf(kamping::as_deserializable(data)),
                  kamping::source(1), kamping::tag(comm.default_tag()));
      }
    } else if (comm.rank() == 1) {
      if (mpi_type_constructor != Type::serialization) {
        comm.recv(kamping::recv_buf(data), kamping::recv_type(type),
                  kamping::recv_count(num_elements), kamping::source(0),
                  kamping::tag(comm.default_tag()));
        comm.send(kamping::send_buf(data), kamping::send_type(type),
                  kamping::send_count(num_elements), kamping::destination(0));
      } else {
        comm.recv(kamping::recv_buf(kamping::as_deserializable(data)),
                  kamping::source(0), kamping::tag(comm.default_tag()));
        comm.send(kamping::send_buf(kamping::as_serialized(data)),
                  kamping::destination(0));
      }
    }
    double end = MPI_Wtime();
    times[i] = end - start;
  }
  return times;
}

auto main(int argc, char* argv[]) -> int {
  kamping::Environment env;
  CLI::App app{"MPI datatype ping-pong benchmark"};
  std::size_t n_reps;
  std::size_t data_size;
  app.add_option("--n_reps", n_reps)->required()->check(CLI::PositiveNumber);
  app.add_option("--data_size", data_size)
      ->required()
      ->check(CLI::PositiveNumber);
  Type mpi_type_constructor;
  app.add_option("--mpi_type_constructor", mpi_type_constructor)
      ->required()
      ->transform(CLI::CheckedTransformer(std::unordered_map<std::string, Type>{
          {"create_struct", Type::create_struct},
          {"pair_as_bytes", Type::pair_as_bytes},
          {"contiguous_type", Type::contiguous_type},
          {"serialization", Type::serialization},
          {"builtin", Type::builtin}}));
  std::string json_output_path = "stdout";
  app.add_option("--json_output_path", json_output_path);
  CLI11_PARSE(app, argc, argv);

  std::unique_ptr<std::ostream> out;
  if (json_output_path == "stdout") {
    out = std::make_unique<std::ostream>(std::cout.rdbuf());
  } else {
    std::ofstream file_output(json_output_path);
    out = std::make_unique<std::ofstream>(std::move(file_output));
  }

  kamping::Communicator comm;

  auto times = benchmark<std::pair<std::int32_t, int64_t>>(
      comm, n_reps, data_size, mpi_type_constructor);
  log_result(times, n_reps, data_size, mpi_type_constructor, "std::pair<int32_t,int64_t>", *out);
  times = benchmark<std::pair<std::int64_t, int64_t>>(
      comm, n_reps, data_size, mpi_type_constructor);
  log_result(times, n_reps, data_size, mpi_type_constructor, "std::pair<int64_t,int64_t>", *out);
  return 0;
}
