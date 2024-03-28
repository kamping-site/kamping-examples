#include <mpi.h>
#include <spdlog/pattern_formatter.h>

class rank_formatter : public spdlog::custom_flag_formatter {
 public:
  void format(const spdlog::details::log_msg &, const std::tm &,
              spdlog::memory_buf_t &dest) override {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto rank_string = std::to_string(rank);
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
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto size_string = std::to_string(size);
    dest.append(size_string);
  }
  std::unique_ptr<custom_flag_formatter> clone() const override {
    return spdlog::details::make_unique<size_formatter>();
  }
};
