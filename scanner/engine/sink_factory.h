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

#include "scanner/api/sink.h"
#include "scanner/util/common.h"

#include <vector>

namespace scanner {

namespace internal {

/**
 * @brief Interface for constructing Sinks at runtime.
 */
class SinkFactory {
 public:
  SinkFactory(const std::string& name, bool variadic_inputs,
              const std::vector<Column>& input_columns, bool per_element_output,
              bool entire_stream_output, const std::string& protobuf_name,
              const std::string& stream_protobuf_name,
              SinkConstructor constructor)
    : name_(name),
      variadic_inputs_(variadic_inputs),
      input_columns_(input_columns),
      per_element_output_(per_element_output),
      entire_stream_output_(entire_stream_output),
      protobuf_name_(protobuf_name),
      stream_protobuf_name_(stream_protobuf_name),
      constructor_(constructor) {}

  const std::string& get_name() const { return name_; }

  const bool variadic_inputs() const { return variadic_inputs_; }

  const std::vector<Column>& input_columns() const { return input_columns_; }

  const bool per_element_output() const { return per_element_output_; }

  const bool entire_stream_output() const { return entire_stream_output_; }

  const std::string& protobuf_name() const { return protobuf_name_; }

  const std::string& stream_protobuf_name() const {
    return stream_protobuf_name_;
  }

  /* @brief Constructs a Sink to be used for writing elements
   */
  Sink* new_instance(const SinkConfig& config) { return constructor_(config); }

 private:
  std::string name_;
  bool variadic_inputs_;
  std::vector<Column> input_columns_;
  bool per_element_output_;
  bool entire_stream_output_;
  std::string protobuf_name_;
  std::string stream_protobuf_name_;
  SinkConstructor constructor_;
};
}
}
