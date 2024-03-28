#include <mpi.h>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <kamping/collectives/allgather.hpp>
#include <kamping/communicator.hpp>
#include <kamping/mpi_datatype.hpp>
#include <mpi/all.hpp>
#include <mpi/third_party/pfr.hpp>
#include <mpl/mpl.hpp>
#include <numeric>
#include <vector>

#include "./mpi_spd_formatters.hpp"
#if defined(KAMPING_EXAMPLES_USE_BOOST)
#include "vector_allgather/boost.hpp"
#endif
#include "vector_allgather/kamping.hpp"
#include "vector_allgather/mpi.hpp"
#include "vector_allgather/mpl.hpp"
#include "vector_allgather/rwth_mpi.hpp"

std::vector<int> generate_input(size_t local_size, MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  return std::vector<int>(local_size, rank);
}

int main() {
  mpl::environment::comm_world();  // MPL implicitely initializes the
                                   // environment on first access to comm_world
  auto formatter = std::make_unique<spdlog::pattern_formatter>();
  formatter->add_flag<rank_formatter>('r');
  formatter->add_flag<size_formatter>('s');
  formatter->set_pattern("[%r/%s] [%^%l%$] %v");
  spdlog::set_formatter(std::move(formatter));

  auto v_local = generate_input(3, MPI_COMM_WORLD);
  {
    auto v_global = mpi::get_whole_vector(v_local, MPI_COMM_WORLD);
    spdlog::info("plain mpi: {}", v_global);
  }
  {
#if defined(KAMPING_EXAMPLES_USE_BOOST)
    auto v_global = boost::get_whole_vector(v_local, MPI_COMM_WORLD);
    spdlog::info("Boost.MPI: {}", v_global);
#endif
  }
  {
    auto v_global = mpl::get_whole_vector(v_local, MPI_COMM_WORLD);
    spdlog::info("MPL: {}", v_global);
  }
  {
    auto v_global = rwth::get_whole_vector(v_local, MPI_COMM_WORLD);
    spdlog::info("RWTH-MPI: {}", v_global);
  }
  {
    auto v_global = kamping::get_whole_vector(v_local, MPI_COMM_WORLD);
    spdlog::info("__MPIng: {}", v_global);
  }
  return 0;
}
