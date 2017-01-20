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

#include "scanner/util/common.h"
#include "scanner/util/profiler.h"

#include <vector>

namespace scanner {
namespace api {

struct EvalInput;

class Evaluator {
public:
  Evaluator(const std::string &name, const std::vector<EvalInput> &inputs,
            char *args, size_t args_size);

  virtual ~Evaluator(){};

protected:
  std::string name_;
  std::vector<EvalInput> inputs_;
  char* args;
};

struct EvalInput {
  Evaluator* evaluator;
  std::vector<std::string> columns;
};

Evaluator* make_input_evaluator(const std::vector<std::string>& columns);

}

///////////////////////////////////////////////////////////////////////////////
/// Implementation Details
class EvaluatorRegistration {
 public:
  EvaluatorRegistration(const EvaluatorBuilder& builder);
};

class EvaluatorBuilder {
 public:
  friend class EvaluatorRegistration;

  EvaluatorBuilder(const std::string &name)
      : name_(name) {}

  EvaluatorBuilder& outputs(const std::vector<std::string>& columns) {
    output_columns_ = columns;
    return *this;
  }

 private:
  std::string name_;
  std::vector<std::string> output_columns_;
};

#define REGISTER_EVALUATOR(name) \
  REGISTER_EVALUATOR_UID(__COUNTER__, name)

#define REGISTER_EVALUATOR_UID(uid, name) \
  static ::scanner::EvaluatorRegistration \
      evaluator_registration_##uid## = EvaluatorBuilder(#name)

}
