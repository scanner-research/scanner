/* Copyright 2017 Carnegie Mellon University
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
#include "scanner/util/common.h"
#include "scanner/util/profiler.h"
#include "scanner/util/storehouse.h"

#include <vector>

namespace scanner {

//! Parameters provided at instantiation of Sink node
struct SinkConfig {
  std::vector<std::string> input_columns;
  std::vector<proto::ColumnType> input_column_types;
  std::vector<u8> args;  //! Byte-string of proto args if given.
  i32 node_id;
  storehouse::StorageConfig* storage_config;
};

/**
 * @brief Interface for reading data in a computation graph.
 *
 * Sink are how Scanner's computation graphs read data from outside the
 * system.
 */
class Sink {
 public:
  Sink(const SinkConfig& config) {}

  virtual ~Sink(){};

  /**
   * @brief Checks if Sink arguments are valid.
   *
   * Only useful if your Sink has its own custom Protobuf arguments.
   */
  virtual void validate(proto::Result* result) { result->set_success(true); }

  /**
   * @brief Called when the Sink is about to process a new stream.
   *
   * @param args
   *        the arguments that were bound to this output for this stream
   *
   */
  virtual void new_stream(const std::vector<u8>& args) {};

  /**
   * @brief Runs the Sink to write elements.
   *
   * @param input_columns
   *        sink input, vector of elements produced by an Op
   *
   */
  virtual void write(const BatchedElements& input_columns) = 0;

  /**
   * @brief When this function returns, the data for all previous 'write'
   *        calls MUST BE durably written.
   */
  virtual void finished() {};

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

class SinkBuilder;

using SinkConstructor =
    std::function<Sink*(const SinkConfig& config)>;

class SinkRegistration {
 public:
  SinkRegistration(const SinkBuilder& builder);
};

class SinkBuilder {
 public:
  friend class SinkRegistration;

  SinkBuilder(const std::string& name, SinkConstructor constructor)
    : name_(name),
      constructor_(constructor),
      variadic_inputs_(false),
      per_element_output_(false),
      entire_stream_output_(false) {}

  SinkBuilder& variadic_inputs() {
    if (input_columns_.size() > 0) {
      LOG(FATAL) << "Sink " << name_ << " cannot have both fixed and variadic "
                 << "inputs";
    }
    variadic_inputs_ = true;
    return *this;
  }

  SinkBuilder& input(const std::string& name,
                   ColumnType type = ColumnType::Bytes) {
    if (variadic_inputs_) {
      LOG(FATAL) << "Sink " << name_ << " cannot have both fixed and variadic "
                 << "inputs";
    }
    input_columns_.push_back(std::make_tuple(name, type));
    return *this;
  }

  SinkBuilder& frame_input(const std::string& name) {
    return input(name, ColumnType::Video);
  }

  SinkBuilder& per_element_output() {
    if (entire_stream_output_) {
      LOG(FATAL) << "Sink " << name_
                 << " cannot specify both per element and entire stream output";
    }
    per_element_output_ = true;
    return *this;
  }

  SinkBuilder& entire_stream_output() {
    LOG(FATAL) << "Entire stream output is not implemented yet.";

    if (per_element_output_) {
      LOG(FATAL) << "Sink " << name_
                 << " cannot specify both per element and entire stream output";
    }
    entire_stream_output_ = true;
    return *this;
  }

  SinkBuilder& protobuf_name(const std::string& name) {
    protobuf_name_ = name;
    return *this;
  }

  SinkBuilder& stream_protobuf_name(const std::string& name) {
    stream_protobuf_name_ = name;
    return *this;
  }

 private:
  std::string name_;
  SinkConstructor constructor_;
  bool variadic_inputs_;
  std::vector<std::tuple<std::string, ColumnType>> input_columns_;
  bool per_element_output_;
  bool entire_stream_output_;
  std::string protobuf_name_;
  std::string stream_protobuf_name_;
};
}

#define REGISTER_SINK(name__, sink__) \
  REGISTER_SINK_HELPER(__COUNTER__, name__, sink__)

#define REGISTER_SINK_HELPER(uid__, name__, sink__)    \
  REGISTER_SINK_UID(uid__, name__, sink__)

#define REGISTER_SINK_UID(uid__, name__, sink__)                              \
  static ::scanner::internal::SinkRegistration sink_registration_##uid__ \
      __attribute__((unused)) = ::scanner::internal::SinkBuilder(          \
          #name__, [](const ::scanner::SinkConfig& config) {               \
            return new sink__(config);                                     \
          })
}
