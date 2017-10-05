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

#include "scanner/api/op.h"
#include "scanner/engine/op_info.h"
#include "scanner/engine/op_registry.h"

namespace scanner {
namespace internal {

OpRegistration::OpRegistration(const OpBuilder& builder) {
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
  std::vector<Column> output_columns;
  i = 0;
  for (auto& name_type : builder.output_columns_) {
    Column col;
    col.set_id(i++);
    col.set_name(std::get<0>(name_type));
    col.set_type(std::get<1>(name_type));
    output_columns.push_back(col);
  }
  bool can_stencil = builder.can_stencil_;
  const std::vector<i32>& stencil = builder.preferred_stencil_;
  bool has_bounded_state = builder.has_bounded_state_;
  i32 warmup = builder.warmup_;
  bool has_unbounded_state = builder.has_unbounded_state_;
  OpInfo* info = new OpInfo(name, variadic_inputs, input_columns,
                            output_columns, can_stencil, stencil,
                            has_bounded_state, warmup, has_unbounded_state);
  OpRegistry* registry = get_op_registry();
  Result result = registry->add_op(name, info);
  if (!result.success()) {
    LOG(WARNING) << "Failed to register op " << name << ": " << result.msg();
  }
}
}
}
