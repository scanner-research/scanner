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

//! Interface for a computation unit implemented by a Kernel.
class Op {
 public:
  Op(const std::string& name, const std::vector<OpInput>& inputs,
     DeviceType device_type, char* args = nullptr, size_t args_size = 0);

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

//! Set of inputs provded to an op in a computation DAG.
class OpInput {
 public:
  OpInput(Op* op, const std::vector<std::string>& columns)
    : op(op), columns(columns) {}

  Op* get_op() const;

  const std::vector<std::string>& get_columns() const;

 private:
  Op* op;
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

  OpBuilder(const std::string& name) : name_(name), variadic_inputs_(false) {}

  OpBuilder& variadic_inputs() {
    if (input_columns_.size() > 0) {
      LOG(FATAL) << "Op " << name_ << " cannot have both fixed and variadic "
                 << "inputs";
    }
    variadic_inputs_ = true;
    return *this;
  }

  OpBuilder& input(const std::string& name,
                   ColumnType type = ColumnType::Other) {
    if (variadic_inputs_) {
      LOG(FATAL) << "Op " << name_ << " cannot have both fixed and variadic "
                 << "inputs";
    }
    Column col;
    col.set_id(input_columns_.size());
    col.set_name(name);
    col.set_type(type);
    input_columns_.push_back(col);
    return *this;
  }

  OpBuilder& frame_input(const std::string& name) {
    return input(name, ColumnType::Video);
  }

  OpBuilder& output(const std::string& name,
                    ColumnType type = ColumnType::Other) {
    Column col;
    col.set_id(input_columns_.size());
    col.set_name(name);
    col.set_type(type);
    output_columns_.push_back(col);
    return *this;
  }

  OpBuilder& frame_output(const std::string& name) {
    return output(name, ColumnType::Video);
  }

 private:
  std::string name_;
  bool variadic_inputs_;
  std::vector<Column> input_columns_;
  std::vector<Column> output_columns_;
};
}

#define REGISTER_OP(name__) REGISTER_OP_HELPER(__COUNTER__, name__)

#define REGISTER_OP_HELPER(uid__, name__) REGISTER_OP_UID(uid__, name__)

#define REGISTER_OP_UID(uid__, name__)                               \
  static ::scanner::internal::OpRegistration op_registration_##uid__ \
    __attribute__((unused)) = ::scanner::internal::OpBuilder(#name__)
}
