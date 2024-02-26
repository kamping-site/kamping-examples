#include <spdlog/pattern_formatter.h>
#include <kamping/communicator.hpp>
#include <kamping/environment.hpp>

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
