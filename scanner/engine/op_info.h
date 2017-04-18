/* Copyright 2016 Carnegie Mellon University
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

#include "scanner/api/op.h"
#include "scanner/util/common.h"

#include <vector>

namespace scanner {
namespace internal {

class OpInfo {
 public:
  OpInfo(const std::string& name, bool variadic_inputs,
         const std::vector<Column>& input_columns,
         const std::vector<Column>& output_columns)
    : name_(name),
      variadic_inputs_(variadic_inputs),
      input_columns_(input_columns),
      output_columns_(output_columns) {}

  const std::string& name() const { return name_; }

  const bool variadic_inputs() const { return variadic_inputs_; }

  const std::vector<Column>& input_columns() const {
    return input_columns_;
  }

  const std::vector<Column>& output_columns() const {
    return output_columns_;
  }

 private:
  std::string name_;
  bool variadic_inputs_;
  std::vector<Column> input_columns_;
  std::vector<Column> output_columns_;
};
}
}
