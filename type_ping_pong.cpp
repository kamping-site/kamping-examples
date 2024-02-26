#include <mpi.h>

#include <CLI/CLI.hpp>
#include <cstddef>
#include <utility>
#include <vector>

enum class Type { create_struct, pair_as_bytes, contiguous_type, builtin };

std::string to_string(Type t) {
  switch (t) {
    case Type::create_struct:
      return "create_struct";
    case Type::pair_as_bytes:
      return "pair_as_bytes";
    case Type::contiguous_type:
      return "contiguous_type";
    case Type::builtin:
      return "builtin";
  }
  return "unknown";
}

auto main(int argc, char* argv[]) -> int {
  MPI_Init(&argc, &argv);
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
          {"builtin", Type::builtin}}));
  std::string json_output_path = "stdout";
  app.add_option("--json_output_path", json_output_path);
  CLI11_PARSE(app, argc, argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  using value_type = int;
  std::vector<value_type> data(data_size);
  MPI_Datatype type;
  int num_elements;
  switch (mpi_type_constructor) {
    case Type::create_struct: {
      // value_type t{};
      // MPI_Aint base;
      // MPI_Get_address(&t, &base);
      // MPI_Aint disp[2];
      // MPI_Get_address(&t.first, &disp[0]);
      // MPI_Get_address(&t.second, &disp[1]);
      // disp[0] = MPI_Aint_diff(disp[0], base);
      // disp[1] = MPI_Aint_diff(disp[1], base);
      // MPI_Datatype types[2] = {MPI_INT, MPI_INT};
      // int blocklens[2] = {1, 1};
      int blocklens = 1;
      MPI_Aint disp = 0;
      MPI_Datatype types = MPI_INT;
      MPI_Type_create_struct(1, &blocklens, &disp, &types, &type);
      // MPI_Type_contiguous(2, MPI_INT, &type);
      MPI_Type_commit(&type);
      num_elements = static_cast<int>(data.size());
      break;
    }
    case Type::pair_as_bytes: {
      MPI_Type_contiguous(sizeof(value_type), MPI_BYTE, &type);
      MPI_Type_commit(&type);
      num_elements = static_cast<int>(data.size());
      break;
    }
    case Type::contiguous_type:
      MPI_Type_contiguous(static_cast<int>(data_size) * sizeof(value_type),
                          MPI_BYTE, &type);
      MPI_Type_commit(&type);
      num_elements = 1;
      break;
    case Type::builtin:
      type = MPI_INT;
      num_elements = static_cast<int>(data.size());
      break;
  }
  std::vector<double> times(n_reps);
  for (std::size_t i = 0; i < n_reps; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if (rank == 0) {
      MPI_Send(data.data(), num_elements, type, 1, 0, MPI_COMM_WORLD);
      MPI_Recv(data.data(), num_elements, type, 1, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    } else if (rank == 1) {
      MPI_Recv(data.data(), num_elements, type, 0, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Send(data.data(), num_elements, type, 0, 0, MPI_COMM_WORLD);
    }
    double end = MPI_Wtime();
    times[i] = end - start;
  }
  if (rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, times.data(), static_cast<int>(n_reps), MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(times.data(), nullptr, static_cast<int>(n_reps), MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
  }
  if (rank == 0) {
    auto& out = std::cout;
    // std::ofstream out(json_output_path);
    double average_time =
        std::accumulate(times.begin(), times.end(), 0.0) / n_reps;
    auto [min_time, max_time] = std::minmax_element(times.begin(), times.end());
    out << "RESULT ";
    out << "n_reps=" << n_reps << " data_size=" << data_size
        << " mpi_type_constructor=" << to_string(mpi_type_constructor) << " ";
    out << "average_time=" << average_time << " min_time=" << *min_time
        << " max_time=" << *max_time << "\n";
  }
  MPI_Type_free(&type);
  MPI_Finalize();
  return 0;
}
