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

Op::Op(const std::string& name, const std::vector<OpInput>& inputs,
       DeviceType device_type, char* args, size_t args_size,
       const std::vector<i32>& stencil, i32 batch_size)
  : name_(name),
    inputs_(inputs),
    type_(device_type),
    args_(args),
    args_size_(args_size),
    stencil_(stencil),
    batch_size_(batch_size) {}

const std::string& Op::get_name() const { return name_; }

const std::vector<OpInput>& Op::get_inputs() const { return inputs_; }

DeviceType Op::get_device_type() const { return type_; }

char* Op::get_args() const { return args_; }

size_t Op::get_args_size() const { return args_size_; }

const std::vector<i32>& Op::get_stencil() const {
  return stencil_;
}

i32 Op::get_batch_size() const {
  return batch_size_;
}


Op* OpInput::get_op() const { return op; }

const std::vector<std::string>& OpInput::get_columns() const { return columns; }

Op* make_input_op(const std::vector<std::string>& columns) {
  OpInput eval_input = {nullptr, columns};
  return new Op("InputTable", {eval_input}, DeviceType::CPU);
}

Op* make_output_op(const std::vector<OpInput>& inputs) {
  return new Op("OutputTable", inputs, DeviceType::CPU);
}

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
  OpInfo* info =
      new OpInfo(name, variadic_inputs, input_columns, output_columns,
                 can_stencil, stencil);
  OpRegistry* registry = get_op_registry();
  registry->add_op(name, info);
}
}
}
