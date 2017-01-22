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
#include "scanner/api/evaluator.h"
#include "scanner/util/common.h"

#include <vector>

namespace scanner {

namespace internal {

/**
 * @brief Interface for constructing evaluators at runtime.
 *
 * Scanner pipelines are composed of a sequence of evaluator factories. A single
 * job may use any number of a given evaluator, so the EvaluatorFactory allows
 * the user to capture configuration information about the evaluator (e.g. batch
 * size of a neural net, device type) and pass that information to each new
 * evaluator instance. The EvaluatorFactory also provides metadata about
 * the inputs and outputs from the evaluator it produces.
 */
class KernelFactory {
 public:
  KernelFactory(const std::string& evaluator_name,
                DeviceType type, i32 max_devices, i32 warmup_size,
                KernelConstructor constructor)
      : evaluator_name_(evaluator_name),
        type_(type), max_devices_(max_devices), warmup_size_(warmup_size),
        constructor_(constructor) {}

  /** Describes the capabilities of the evaluators the factory produces. */
  DeviceType get_device_type();

  i32 get_max_devices();

  i32 get_warmup_size();

  /* @brief Constructs a kernel to be used for processing rows of data.
   */
  Kernel* new_instance(const Kernel::Config& config) {
    return constructor_(config);
  }

 private:
  std::string evaluator_name_;
  DeviceType type_;
  i32 max_devices_;
  i32 warmup_size_;
  KernelConstructor constructor_;
};

}
}
