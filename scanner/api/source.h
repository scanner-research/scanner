/* Copyright 2018 Carnegie Mellon University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "scanner/api/kernel.h"
#include "scanner/api/enumerator.h"
#include "scanner/util/common.h"
#include "scanner/util/profiler.h"

#include <vector>

namespace scanner {

//! Parameters provided at instantiation of Source node
struct SourceConfig {
  std::vector<std::string> output_columns;
  std::vector<proto::ColumnType> output_column_types;
  std::vector<u8> args;  //! Byte-string of proto args if given.
  i32 node_id;
};

/**
 * @brief Interface for reading data in a computation graph.
 *
 * Sources are how Scanner's computation graphs read data from outside the
 * system.
 */
class Source {
 public:
  Source(const SourceConfig& config) {}

  virtual ~Source(){};

  /**
   * @brief Checks if Source arguments are valid.
   *
   * Only useful if your Source has its own custom Protobuf arguments.
   */
  virtual void validate(proto::Result* result) { result->set_success(true); }

  /**
   * @brief Runs the Source to generate input elements.
   *
   * @param args
   *        vector of ElementArgs produced by an Enumerator, where each
   *        ElementArgs describes which elements to read.
   * @param output_columns
   *        source output, vector of elements, where each element was produced
   *        using the provided ElementArgs
   *
   * Number of output columns must be non-zero.
   */
  virtual void read(const std::vector<ElementArgs>& args,
                    BatchedElements& output_columns) = 0;

  /**
   * @brief For internal use
   **/
  virtual void set_profiler(Profiler* profiler) { profiler_ = profiler; }

  /**
   * The profiler allows an op to save profiling data for later
   * visualization. It is not guaranteed to be non-null, so check before use.
   */
  Profiler* profiler_ = nullptr;
};

///////////////////////////////////////////////////////////////////////////////
/// Implementation Details
namespace internal {

class SourceBuilder;

using SourceConstructor =
    std::function<Source*(const SourceConfig& config)>;

class SourceRegistration {
 public:
  SourceRegistration(const SourceBuilder& builder);
};

class SourceBuilder {
 public:
  friend class SourceRegistration;

  SourceBuilder(const std::string& name, SourceConstructor constructor)
    : name_(name),
      constructor_(constructor) {}

  SourceBuilder& output(const std::string& name,
                    ColumnType type = ColumnType::Other) {
    if (output_columns_.size() > 0) {
      LOG(FATAL) << "Sources can only have one output column.";
    }
    output_columns_.push_back(std::make_tuple(name, type));
    return *this;
  }

  SourceBuilder& frame_output(const std::string& name) {
    return output(name, ColumnType::Video);
  }

 private:
  std::string name_;
  SourceConstructor constructor_;
  std::vector<std::tuple<std::string, ColumnType>> output_columns_;
};
}

#define REGISTER_SOURCE(name__, source__)              \
  REGISTER_SOURCE_HELPER(__COUNTER__, name__, source__)

#define REGISTER_SOURCE_HELPER(uid__, name__, source__) \
  REGISTER_SOURCE_UID(uid__, name__, source__)

#define REGISTER_SOURCE_UID(uid__, name__, source__)                         \
  static ::scanner::internal::SourceRegistration source_registration_##uid__ \
      __attribute__((unused)) = ::scanner::internal::SourceBuilder(          \
          #name__, [](const ::scanner::SourceConfig& config) {               \
            return new source__(config);                                     \
          })
}
