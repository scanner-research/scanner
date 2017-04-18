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

#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/common.h"

#include <vector>

namespace scanner {

namespace internal {

/**
 * @brief Interface for constructing ops at runtime.
 *
 * Scanner pipelines are composed of a sequence of op factories. A single
 * job may use any number of a given op, so the OpFactory allows
 * the user to capture configuration information about the op (e.g. batch
 * size of a neural net, device type) and pass that information to each new
 * op instance. The OpFactory also provides metadata about
 * the inputs and outputs from the op it produces.
 */
class KernelFactory {
 public:
  KernelFactory(const std::string& op_name, DeviceType type, i32 max_devices,
                i32 warmup_size, KernelConstructor constructor)
      : op_name_(op_name),
        type_(type),
        max_devices_(max_devices),
        warmup_size_(warmup_size),
        constructor_(constructor) {}

  const std::string& get_op_name() const { return op_name_; }

  /** Describes the capabilities of the ops the factory produces. */
  DeviceType get_device_type() const { return type_; }

  i32 get_max_devices() const { return max_devices_; }

  i32 get_warmup_size() const { return warmup_size_; }

  /* @brief Constructs a kernel to be used for processing elements of data.
   */
  Kernel* new_instance(const Kernel::Config& config) {
    return constructor_(config);
  }

 private:
  std::string op_name_;
  DeviceType type_;
  i32 max_devices_;
  i32 warmup_size_;
  KernelConstructor constructor_;
};
}
}
