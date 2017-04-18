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
       DeviceType device_type, char* args, size_t args_size)
  : name_(name),
    inputs_(inputs),
    type_(device_type),
    args_(args),
    args_size_(args_size) {}

const std::string& Op::get_name() const { return name_; }

const std::vector<OpInput>& Op::get_inputs() const { return inputs_; }

DeviceType Op::get_device_type() const { return type_; }

char* Op::get_args() const { return args_; }

size_t Op::get_args_size() const { return args_size_; }

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
  const std::vector<Column>& input_columns = builder.input_columns_;
  const std::vector<Column>& output_columns = builder.output_columns_;
  OpInfo* info =
    new OpInfo(name, variadic_inputs, input_columns, output_columns);
  OpRegistry* registry = get_op_registry();
  registry->add_op(name, info);
}
}
}
