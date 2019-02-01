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
      : name_(name), variadic_inputs_(false), can_stencil_(false),
        has_bounded_state_(false), warmup_(0), has_unbounded_state_(false) {}

  OpBuilder& variadic_inputs() {
    if (input_columns_.size() > 0) {
      LOG(FATAL) << "Op " << name_ << " cannot have both fixed and variadic "
                 << "inputs";
    }
    variadic_inputs_ = true;
    return *this;
  }

  OpBuilder& input(const std::string& name,
                   ColumnType type = ColumnType::Bytes) {
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
                    ColumnType type = ColumnType::Bytes,
                    std::string type_name = "") {
    output_columns_.push_back(std::make_tuple(name, type, type_name));
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

  OpBuilder& bounded_state(i32 warmup = 0) {
    if (has_unbounded_state_) {
      LOG(FATAL) << "Attempted to specify Op " << name_
                 << " has bounded state but Op was already declared to have "
                    "unbounded state.";
    }
    has_bounded_state_ = true;
    warmup_ = warmup;
    return *this;
  }

  OpBuilder& unbounded_state() {
    if (has_bounded_state_) {
      LOG(FATAL) << "Attempted to specify Op " << name_
                 << " has unbounded state but Op was already declared to have "
                    "bounded state.";
    }
    has_unbounded_state_ = true;
    return *this;
  }

  OpBuilder& protobuf_name(std::string protobuf_name) {
    protobuf_name_ = protobuf_name;
    return *this;
  }

 private:
  std::string name_;
  bool variadic_inputs_;
  std::vector<std::tuple<std::string, ColumnType>> input_columns_;
  std::vector<std::tuple<std::string, ColumnType, std::string>> output_columns_;
  bool can_stencil_;
  std::vector<int> preferred_stencil_ = {0};
  bool has_bounded_state_;
  i32 warmup_;
  bool has_unbounded_state_;
  std::string protobuf_name_;
};
}

#define REGISTER_OP(name__) REGISTER_OP_HELPER(__COUNTER__, name__)

#define REGISTER_OP_HELPER(uid__, name__) REGISTER_OP_UID(uid__, name__)

#define REGISTER_OP_UID(uid__, name__)                               \
  static ::scanner::internal::OpRegistration op_registration_##uid__ \
      __attribute__((unused)) = ::scanner::internal::OpBuilder(#name__)
}
