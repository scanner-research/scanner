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

#include "scanner/engine/sink_registry.h"

namespace scanner {
namespace internal {

Result SinkRegistry::add_sink(const std::string& name,
                                SinkFactory* factory) {
  Result result;
  result.set_success(true);
  if (factories_.count(name) > 0) {
    RESULT_ERROR(&result, "Attempted to re-register Sink %s", name.c_str());
    return result;
  }
  if (factory->input_columns().empty() && !factory->variadic_inputs()) {
    RESULT_ERROR(&result,
                 "Attempted to register Sink %s with empty input columns",
                 name.c_str());
    return result;
  }

  if (factory->per_element_output() && factory->entire_stream_output()) {
    RESULT_ERROR(&result,
                 "Attempted to register Sink %s with both per-element and "
                 "entire stream output. Specify only one.",
                 name.c_str());
    return result;
  }

  if (!factory->per_element_output() && !factory->entire_stream_output()) {
    RESULT_ERROR(&result,
                 "Attempted to register Sink %s with neither per-element or "
                 "entire stream output. One must be specified.",
                 name.c_str());
    return result;
  }

  factories_.insert({name, factory});

  return result;
}

bool SinkRegistry::has_sink(const std::string& name) {
  return factories_.count(name) > 0;
}

SinkFactory* SinkRegistry::get_sink(const std::string& name) {
  return factories_.at(name);
}

SinkRegistry* get_sink_registry() {
  static SinkRegistry* registry = new SinkRegistry;
  return registry;
}
}
}
