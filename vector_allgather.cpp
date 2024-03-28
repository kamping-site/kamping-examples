#include <mpi.h>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <boost/mpi.hpp>
#include <boost/mpi/collectives/all_gatherv.hpp>
#include <kamping/collectives/allgather.hpp>
#include <kamping/communicator.hpp>
#include <kamping/mpi_datatype.hpp>
#include <mpi/all.hpp>
#include <mpi/third_party/pfr.hpp>
#include <mpl/mpl.hpp>
#include <numeric>
#include <vector>

#include "./mpi_spd_formatters.hpp"

namespace mpi {
template <typename T>
std::vector<T> get_whole_vector(std::vector<T> const& v_local, MPI_Comm comm) {
  int size;
  int rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  std::vector<int> rc(size), rd(size);
  rc[rank] = v_local.size();
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rc.data(), 1, MPI_INT,
                comm);
  std::exclusive_scan(rc.begin(), rc.end(), rd.begin(), 0);
  std::vector<T> v_global(rd.back() + rc.back());
  MPI_Allgatherv(v_local.data(), v_local.size(), kamping::mpi_datatype<T>(),
                 v_global.data(), rc.data(), rd.data(),
                 kamping::mpi_datatype<T>(), comm);
  return v_global;
}
}  // namespace mpi
namespace boost {
template <typename T>
std::vector<T> get_whole_vector(std::vector<T> const& v_local, MPI_Comm comm_) {
  boost::mpi::communicator comm{comm_, mpi::comm_attach};
  std::vector<T> v_global;
  std::vector<int> sizes;
  boost::mpi::all_gather(comm, static_cast<int>(v_local.size()), sizes);
  boost::mpi::all_gatherv(comm, v_local, v_global, sizes);
  return v_global;
}
}  // namespace boost
namespace mpl {
template <typename T>
std::vector<T> get_whole_vector(std::vector<T> const& v_local, MPI_Comm) {
  auto const& comm = mpl::environment::comm_world();
  std::vector<int> rc(comm.size());
  comm.allgather(static_cast<int>(v_local.size()), rc.data());
  mpl::layouts<T> recvls;
  int recv_displ = 0;
  for (int rank = 0; rank < comm.size(); ++rank) {
    recvls.push_back(mpl::indexed_layout<T>({{rc[rank], recv_displ}}));
    recv_displ += rc[rank];
  }
  std::vector<T> v_global(recv_displ);
  mpl::contiguous_layout<T> sendl(v_local.size());
  comm.allgatherv(v_local.data(), sendl, v_global.data(), recvls);
  return v_global;
}

}  // namespace mpl
namespace rwth {
template <typename T>
std::vector<T> get_whole_vector(std::vector<T> const& v_local, MPI_Comm comm_) {
  mpi::communicator comm(comm_);
  // std::vector<T> v_global = v_local;
  // // this requires manual patching of the code due to ambiguous overloads and
  // // does not properly resize the buffer
  // comm.all_gather_varying(v_global);
  // return v_global;
  std::vector<int> rc(static_cast<size_t>(comm.size()));
  comm.all_gather(static_cast<int>(v_local.size()), rc);
  std::vector<T> v_global;
  comm.all_gather_varying(v_local, v_global, rc, true);
  return v_global;
}
}  // namespace rwth
namespace __mping {
template <typename T>
std::vector<T> get_whole_vector(std::vector<T> const& v_local, MPI_Comm comm_) {
  using namespace kamping;
  kamping::Communicator comm(comm_);
  return comm.allgatherv(send_buf(v_local));
}
}  // namespace __mping

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
    auto v_global = boost::get_whole_vector(v_local, MPI_COMM_WORLD);
    spdlog::info("Boost.MPI: {}", v_global);
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
    auto v_global = __mping::get_whole_vector(v_local, MPI_COMM_WORLD);
    spdlog::info("__MPIng: {}", v_global);
  }
  return 0;
}
