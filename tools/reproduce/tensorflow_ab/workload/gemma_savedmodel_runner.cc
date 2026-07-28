// SPDX-License-Identifier: MIT

#include <charconv>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

namespace {

struct Feed {
  std::string name;
  tensorflow::DataType dtype;
  tensorflow::TensorShape shape;
  std::string path;
};

std::vector<std::string> Split(const std::string& value, char delimiter) {
  std::stringstream stream(value);
  std::vector<std::string> result;
  std::string part;
  while (std::getline(stream, part, delimiter)) result.push_back(part);
  return result;
}

tensorflow::Status ParseInt(const std::string& value, int* output) {
  int parsed = 0;
  const char* begin = value.data();
  const char* end = begin + value.size();
  const auto result = std::from_chars(begin, end, parsed);
  if (result.ec != std::errc() || result.ptr != end)
    return tensorflow::errors::InvalidArgument("invalid integer: ", value);
  *output = parsed;
  return tensorflow::OkStatus();
}

tensorflow::Status ParseDType(const std::string& value,
                              tensorflow::DataType* dtype) {
  if (value == "int32")
    *dtype = tensorflow::DT_INT32;
  else if (value == "int64")
    *dtype = tensorflow::DT_INT64;
  else if (value == "float32")
    *dtype = tensorflow::DT_FLOAT;
  else
    return tensorflow::errors::InvalidArgument("unsupported dtype: ", value);
  return tensorflow::OkStatus();
}

tensorflow::Status ParseFeed(const std::string& value, Feed* feed) {
  const auto parts = Split(value, ',');
  if (parts.size() != 4)
    return tensorflow::errors::InvalidArgument(
        "feed must be NAME,DTYPE,SHAPE,FILE");
  feed->name = parts[0];
  TF_RETURN_IF_ERROR(ParseDType(parts[1], &feed->dtype));
  for (const auto& dimension : Split(parts[2], 'x')) {
    int size = 0;
    TF_RETURN_IF_ERROR(ParseInt(dimension, &size));
    feed->shape.AddDim(size);
  }
  feed->path = parts[3];
  return tensorflow::OkStatus();
}

tensorflow::Status LoadFeed(
    const Feed& feed,
    std::pair<std::string, tensorflow::Tensor>* output) {
  tensorflow::Tensor tensor(feed.dtype, feed.shape);
  std::ifstream input(feed.path, std::ios::binary | std::ios::ate);
  if (!input) return tensorflow::errors::NotFound(feed.path);
  const auto size = input.tellg();
  if (size != static_cast<std::streamoff>(tensor.TotalBytes()))
    return tensorflow::errors::InvalidArgument(
        "feed byte size mismatch: ", feed.name);
  input.seekg(0);
  input.read(
      const_cast<char*>(tensor.tensor_data().data()), tensor.TotalBytes());
  if (!input)
    return tensorflow::errors::DataLoss("cannot read feed: ", feed.path);
  *output = {feed.name, std::move(tensor)};
  return tensorflow::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  std::string model;
  std::string output_dir;
  std::vector<Feed> feeds;
  std::vector<std::string> fetches;
  int warmup = 2;
  int steps = 10;
  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    if (argument.rfind("--model=", 0) == 0)
      model = argument.substr(8);
    else if (argument.rfind("--output-dir=", 0) == 0)
      output_dir = argument.substr(13);
    else if (argument.rfind("--feed=", 0) == 0) {
      Feed feed;
      auto status = ParseFeed(argument.substr(7), &feed);
      if (!status.ok()) {
        std::cerr << status << "\n";
        return 2;
      }
      feeds.push_back(std::move(feed));
    } else if (argument.rfind("--fetch=", 0) == 0) {
      fetches.push_back(argument.substr(8));
    } else if (argument.rfind("--warmup=", 0) == 0) {
      if (!ParseInt(argument.substr(9), &warmup).ok()) return 2;
    } else if (argument.rfind("--steps=", 0) == 0) {
      if (!ParseInt(argument.substr(8), &steps).ok()) return 2;
    } else {
      std::cerr << "unknown argument: " << argument << "\n";
      return 2;
    }
  }
  if (model.empty() || output_dir.empty() || feeds.empty() ||
      fetches.empty()) {
    std::cerr << "usage: --model=DIR --feed=NAME,DTYPE,SHAPE,FILE "
                 "--fetch=NAME --output-dir=DIR [--warmup=N] "
                 "[--steps=N]\n";
    return 2;
  }

  tensorflow::SavedModelBundle bundle;
  tensorflow::SessionOptions session_options;
  tensorflow::RunOptions run_options;
  auto status = tensorflow::LoadSavedModel(
      session_options, run_options, model,
      {tensorflow::kSavedModelTagServe}, &bundle);
  if (!status.ok()) {
    std::cerr << status << "\n";
    return 1;
  }

  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
  for (const auto& feed : feeds) {
    std::pair<std::string, tensorflow::Tensor> loaded;
    status = LoadFeed(feed, &loaded);
    if (!status.ok()) {
      std::cerr << status << "\n";
      return 1;
    }
    inputs.push_back(std::move(loaded));
  }
  std::vector<tensorflow::Tensor> outputs;
  for (int i = 0; i < warmup; ++i) {
    status = bundle.session->Run(inputs, fetches, {}, &outputs);
    if (!status.ok()) {
      std::cerr << status << "\n";
      return 1;
    }
  }
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < steps; ++i) {
    status = bundle.session->Run(inputs, fetches, {}, &outputs);
    if (!status.ok()) {
      std::cerr << status << "\n";
      return 1;
    }
  }
  const auto elapsed =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - start)
          .count();
  status = tensorflow::Env::Default()->RecursivelyCreateDir(output_dir);
  if (!status.ok()) {
    std::cerr << status << "\n";
    return 1;
  }
  std::cout << "steps=" << steps << " seconds=" << elapsed
            << " steps_per_second=" << steps / elapsed << "\n";
  for (size_t i = 0; i < outputs.size(); ++i) {
    const std::string path =
        output_dir + "/output_" + std::to_string(i) + ".bin";
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
      std::cerr << "cannot create output: " << path << "\n";
      return 1;
    }
    const auto bytes = outputs[i].tensor_data();
    file.write(bytes.data(), bytes.size());
    if (!file) {
      std::cerr << "cannot write output: " << path << "\n";
      return 1;
    }
    std::cout << "output[" << i << "] dtype=" << outputs[i].dtype()
              << " shape=" << outputs[i].shape().DebugString()
              << " bytes=" << bytes.size() << " path=" << path << "\n";
  }
  return 0;
}
