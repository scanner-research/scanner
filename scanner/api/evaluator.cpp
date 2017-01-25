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

#include "scanner/api/evaluator.h"
#include "scanner/engine/evaluator_info.h"
#include "scanner/engine/evaluator_registry.h"

namespace scanner {

Evaluator::Evaluator(const std::string &name,
                     const std::vector<EvalInput> &inputs,
                     DeviceType device_type, char *args, size_t args_size)
    : name_(name), inputs_(inputs), type_(device_type), args_(args),
      args_size_(args_size) {}

const std::string& Evaluator::get_name() const {
  return name_;
}

const std::vector<EvalInput>& Evaluator::get_inputs() const {
  return inputs_;
}

DeviceType Evaluator::get_device_type() const {
  return type_;
}

char* Evaluator::get_args() const {
  return args_;
}

size_t Evaluator::get_args_size() const {
  return args_size_;
}

Evaluator* EvalInput::get_evaluator() const {
  return evaluator;
}

const std::vector<std::string>& EvalInput::get_columns() const {
  return columns;
}

Evaluator* make_input_evaluator(const std::vector<std::string>& columns) {
  EvalInput eval_input = {nullptr, columns};
  return new Evaluator("InputTable", {eval_input}, DeviceType::CPU);
}

Evaluator* make_output_evaluator(const std::vector<EvalInput>& inputs) {
  return new Evaluator("OutputTable", inputs, DeviceType::CPU);
}

namespace internal {

EvaluatorRegistration::EvaluatorRegistration(const EvaluatorBuilder& builder) {
  const std::string &name = builder.name_;
  const std::vector<std::string>& columns = builder.output_columns_;
  EvaluatorInfo* info = new EvaluatorInfo(name, columns);
  EvaluatorRegistry *registry = get_evaluator_registry();
  registry->add_evaluator(name, info);
}

}

}
