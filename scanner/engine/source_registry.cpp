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

#include "scanner/engine/source_registry.h"

namespace scanner {
namespace internal {

Result SourceRegistry::add_source(const std::string& name,
                                SourceFactory* factory) {
  Result result;
  result.set_success(true);
  if (factories_.count(name) > 0) {
    RESULT_ERROR(&result, "Attempted to re-register Source %s", name.c_str());
    return result;
  }
  if (factory->output_columns().empty()) {
    RESULT_ERROR(&result,
                 "Attempted to register op %s with empty output columns",
                 name.c_str());
    return result;
  }

  factories_.insert({name, factory});

  return result;
}

bool SourceRegistry::has_source(const std::string& name) {
  return factories_.count(name) > 0;
}

SourceFactory* SourceRegistry::get_source(const std::string& name) {
  return factories_.at(name);
}

SourceRegistry* get_source_registry() {
  static SourceRegistry* registry = new SourceRegistry;
  return registry;
}
}
}
