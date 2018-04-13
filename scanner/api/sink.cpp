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

#include "scanner/api/sink.h"
#include "scanner/engine/sink_factory.h"
#include "scanner/engine/sink_registry.h"

namespace scanner {
namespace internal {

SinkRegistration::SinkRegistration(const SinkBuilder& builder) {
  const std::string& name = builder.name_;
  const bool variadic_inputs = builder.variadic_inputs_;
  std::vector<Column> input_columns;
  size_t i = 0;
  for (auto& name_type : builder.input_columns_) {
    Column col;
    col.set_id(i++);
    col.set_name(std::get<0>(name_type));
    col.set_type(std::get<1>(name_type));
    input_columns.push_back(col);
  }
  bool per_element_output = builder.per_element_output_;
  bool entire_stream_output = builder.entire_stream_output_;

  SinkConstructor constructor = builder.constructor_;
  internal::SinkFactory* factory = new internal::SinkFactory(
      name, variadic_inputs, input_columns, per_element_output,
      entire_stream_output, builder.protobuf_name_,
      builder.stream_protobuf_name_, constructor);
  SinkRegistry* registry = get_sink_registry();
  Result result = registry->add_sink(name, factory);
  if (!result.success()) {
    LOG(WARNING) << "Failed to register sink " << name << ": " << result.msg();
  }
}
}
}
