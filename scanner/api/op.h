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
     DeviceType device_type, char* args = nullptr, size_t args_size = 0,
     const std::vector<i32>& stencil = {}, i32 batch_size = -1);

  virtual ~Op(){};

  const std::string& get_name() const;

  const std::vector<OpInput>& get_inputs() const;

  DeviceType get_device_type() const;

  char* get_args() const;

  size_t get_args_size() const;

  const std::vector<i32>& get_stencil() const;

  i32 get_batch_size() const;

 protected:
  std::string name_;
  std::vector<OpInput> inputs_;
  DeviceType type_;
  std::vector<i32> stencil_;
  i32 batch_size_;
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

  OpBuilder(const std::string& name)
    : name_(name), variadic_inputs_(false), can_stencil_(false) {}

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
    input_columns_.push_back(std::make_tuple(name, type));
    return *this;
  }

  OpBuilder& frame_input(const std::string& name) {
    return input(name, ColumnType::Video);
  }

  OpBuilder& output(const std::string& name,
                    ColumnType type = ColumnType::Other) {
    output_columns_.push_back(std::make_tuple(name, type));
    return *this;
  }

  OpBuilder& frame_output(const std::string& name) {
    return output(name, ColumnType::Video);
  }

  OpBuilder& stencil(const std::vector<int>& stencil = {0}) {
    can_stencil_ = true;
    preferred_stencil_ = stencil;
    return *this;
  }

 private:
  std::string name_;
  bool variadic_inputs_;
  std::vector<std::tuple<std::string, ColumnType>> input_columns_;
  std::vector<std::tuple<std::string, ColumnType>> output_columns_;
  bool can_stencil_;
  std::vector<int> preferred_stencil_;
};
}

#define REGISTER_OP(name__) REGISTER_OP_HELPER(__COUNTER__, name__)

#define REGISTER_OP_HELPER(uid__, name__) REGISTER_OP_UID(uid__, name__)

#define REGISTER_OP_UID(uid__, name__)                               \
  static ::scanner::internal::OpRegistration op_registration_##uid__ \
      __attribute__((unused)) = ::scanner::internal::OpBuilder(#name__)
}
