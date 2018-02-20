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

#include "scanner/api/source.h"
#include "scanner/util/common.h"

#include <vector>

namespace scanner {

namespace internal {

/**
 * @brief Interface for constructing Sources at runtime.
 */
class SourceFactory {
 public:
  SourceFactory(const std::string& name,
                const std::vector<Column>& output_columns,
                SourceConstructor constructor)
    : name_(name),
      output_columns_(output_columns),
      constructor_(constructor) {}

  const std::string& get_name() const { return name_; }

  const std::vector<Column>& output_columns() const { return output_columns_; }

  /* @brief Constructs a source to be used for reading elements
   */
  Source* new_instance(const SourceConfig& config) {
    return constructor_(config);
  }

 private:
  std::string name_;
  std::vector<Column> output_columns_;
  SourceConstructor constructor_;
};
}
}
