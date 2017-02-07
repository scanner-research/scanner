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

struct OpInput;

class Op {
public:
  Op(const std::string &name, const std::vector<OpInput> &inputs,
            DeviceType device_type,
            char *args = nullptr, size_t args_size = 0);

  virtual ~Op(){};

  const std::string& get_name() const;

  const std::vector<OpInput>& get_inputs() const;

  DeviceType get_device_type() const;

  char* get_args() const;

  size_t get_args_size() const;

protected:
  std::string name_;
  std::vector<OpInput> inputs_;
  DeviceType type_;
  char* args_;
  size_t args_size_;
};

class OpInput {
public:
  OpInput(Op *op, const std::vector<std::string> &columns)
      : op(op), columns(columns) {}

  Op* get_op() const;

  const std::vector<std::string>& get_columns() const;

private:
  Op *op;
  std::vector<std::string> columns;
};

Op* make_input_op(const std::vector<std::string>& columns);

Op* make_output_op(const std::vector<OpInput>& inputs);

///////////////////////////////////////////////////////////////////////////////
/// Implementation Details
namespace internal {

class OpBuilder;

class OpRegistration {
 public:
  OpRegistration(const OpBuilder& builder);
};

class OpBuilder {
 public:
  friend class OpRegistration;

  OpBuilder(const std::string &name)
      : name_(name) {}

  OpBuilder& outputs(const std::vector<std::string>& columns) {
    output_columns_ = columns;
    return *this;
  }

 private:
  std::string name_;
  std::vector<std::string> output_columns_;
};
}

#define REGISTER_OP(name__) \
  REGISTER_OP_UID(__COUNTER__, name__)

#define REGISTER_OP_UID(uid__, name__)                                  \
  static ::scanner::internal::OpRegistration                            \
      op_registration_##uid__ __attribute__((unused)) =                 \
          ::scanner::internal::OpBuilder(#name__)
}
