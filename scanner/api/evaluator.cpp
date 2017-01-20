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

#include "scanner/api/evaluator.h"

namespace scanner {

Evaluator(const std::string &name, const std::vector<EvalInput> &inputs,
          char *args)
    : name_(name), inputs_(inputs), args_(args) {}

Evaluator* make_input_evaluator(const std::vector<std::string>& columns) {
  return new Evaluator("Table", {EvalInput(nullptr, columns)}, nullptr, 0);
}

EvaluatorRegistration::EvaluatorRegistration(const EvaluatorBuilder& builder) {
  const std::string &name = builder.name_;
  const std::vector<std::string>& columns = builder.output_columns_;
  EvaluatorInfo* info = new EvaluatorInfo(name, columns);
  EvaluatorRegistry *registry = get_evaluator_registry();
  registry->add_evaluator(name, info);
}

}
