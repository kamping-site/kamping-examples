#pragma once
#include <algorithm>
#include <cstddef>
#include <vector>
template <typename T>
void pick_splitters(size_t num_splitters, size_t oversampling_ratio,
                    std::vector<T> &global_samples) {
  std::sort(global_samples.begin(), global_samples.end());
  for (size_t i = 0; i < num_splitters; i++) {
    global_samples[i] = global_samples[oversampling_ratio * (i + 1)];
  }
  global_samples.resize(num_splitters);
}

template <typename T>
auto build_buckets(std::vector<T> &data, std::vector<T> &splitters)
    -> std::vector<std::vector<T>> {
  std::vector<std::vector<T>> buckets(splitters.size() + 1);
  for (auto &element : data) {
    const auto bound =
        std::upper_bound(splitters.begin(), splitters.end(), element);
    buckets[bound - splitters.begin()].push_back(element);
  }
  data.clear();
  return buckets;
}
