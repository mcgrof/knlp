// SPDX-License-Identifier: MIT
// Binary I/O adapter only. FDE construction is the unmodified Google library.
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "sketching/point_cloud/fixed_dimensional_encoding.h"

namespace {
void Read(char* data, std::size_t size) {
  if (!std::cin.read(data, size)) throw std::runtime_error("truncated input");
}
uint32_t ReadU32() {
  uint32_t value;
  Read(reinterpret_cast<char*>(&value), sizeof(value));
  return value;
}
void WriteU32(uint32_t value) {
  std::cout.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
int Integer(const std::map<std::string, std::string>& args,
            const std::string& key, int fallback) {
  auto it = args.find(key);
  if (it == args.end()) return fallback;
  std::size_t end;
  long value = std::stol(it->second, &end);
  if (end != it->second.size() || value < 0 ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error("invalid integer for " + key);
  }
  return static_cast<int>(value);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if constexpr (std::endian::native != std::endian::little) {
      throw std::runtime_error("binary adapter requires a little-endian host");
    }
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 >= argc) throw std::runtime_error("missing option value");
      const std::string key(argv[i]);
      if (key != "--role" && key != "--repetitions" && key != "--k-sim" &&
          key != "--seed" && key != "--fill-empty" &&
          key != "--projection-dim" && key != "--final-dim") {
        throw std::runtime_error("unknown option: " + key);
      }
      if (!args.emplace(key, argv[i + 1]).second) {
        throw std::runtime_error("duplicate option: " + key);
      }
    }
    const std::string role = args.at("--role");
    if (role != "query" && role != "document") {
      throw std::runtime_error("role must be query or document");
    }
    char magic[8];
    Read(magic, sizeof(magic));
    if (std::memcmp(magic, "GFDEv1\0\0", 8)) {
      throw std::runtime_error("invalid input magic");
    }
    const uint32_t count = ReadU32();
    const uint32_t dimension = ReadU32();
    if (!count || !dimension || dimension > 1000000) {
      throw std::runtime_error("invalid cloud count or input dimension");
    }
    const int repetitions = Integer(args, "--repetitions", 5);
    const int k_sim = Integer(args, "--k-sim", 4);
    const int seed = Integer(args, "--seed", 1);
    const int projection = Integer(args, "--projection-dim", 0);
    const int final_dimension = Integer(args, "--final-dim", 0);
    const int fill = Integer(args, "--fill-empty", 1);
    if (repetitions < 1 || k_sim > 20 || fill > 1 || final_dimension > 100000000 ||
        seed > std::numeric_limits<int>::max() - repetitions) {
      throw std::runtime_error("invalid FDE configuration");
    }
    const uint64_t raw_dimension = uint64_t(repetitions) * (1ULL << k_sim) *
                                   (projection ? projection : dimension);
    if (raw_dimension > 100000000ULL) {
      throw std::runtime_error("intermediate FDE exceeds 100 million coordinates");
    }
    const uint32_t output_dimension =
        final_dimension ? final_dimension : raw_dimension;
    graph_mining::FixedDimensionalEncodingConfig config;
    config.set_dimension(dimension);
    config.set_num_repetitions(repetitions);
    config.set_num_simhash_projections(k_sim);
    config.set_seed(seed);
    config.set_encoding_type(
        role == "query"
            ? graph_mining::FixedDimensionalEncodingConfig::DEFAULT_SUM
            : graph_mining::FixedDimensionalEncodingConfig::AVERAGE);
    config.set_fill_empty_partitions(role == "document" && fill);
    if (projection) {
      config.set_projection_type(
          graph_mining::FixedDimensionalEncodingConfig::AMS_SKETCH);
      config.set_projection_dimension(projection);
    }
    if (final_dimension) config.set_final_projection_dimension(final_dimension);
    std::cout.write("GFDEo1\0\0", 8);
    WriteU32(count);
    WriteU32(output_dimension);
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t tokens = ReadU32();
      const uint64_t size = uint64_t(tokens) * dimension;
      if (!tokens || size > 100000000ULL) {
        throw std::runtime_error("empty or oversized point cloud");
      }
      std::vector<float> cloud(size);
      Read(reinterpret_cast<char*>(cloud.data()), size * sizeof(float));
      for (float v : cloud) {
        if (!std::isfinite(v)) throw std::runtime_error("nonfinite input");
      }
      auto result = graph_mining::GenerateFixedDimensionalEncoding(cloud, config);
      if (!result.ok()) throw std::runtime_error(result.status().ToString());
      if (result->size() != output_dimension) {
        throw std::runtime_error("upstream output dimension mismatch");
      }
      for (float v : *result) {
        if (!std::isfinite(v)) throw std::runtime_error("nonfinite output");
      }
      std::cout.write(reinterpret_cast<const char*>(result->data()),
                      result->size() * sizeof(float));
    }
    if (std::cin.peek() != std::char_traits<char>::eof()) {
      throw std::runtime_error("unexpected trailing input");
    }
    if (!std::cout) throw std::runtime_error("failed to write output");
  } catch (const std::exception& error) {
    std::cerr << "google_fde: " << error.what() << '\n';
    return 2;
  }
}
