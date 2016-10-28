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

#include "scanner/eval/evaluator.h"
#include "scanner/util/common.h"

#include <vector>

namespace scanner {

struct EvaluatorCapabilities {
  static const i32 UnlimitedDevices = 0;

  DeviceType device_type;
  i32 max_devices;
  i32 warmup_size = 0;
  bool can_overlap = false;
};

struct EvaluatorConfig {
  DeviceType device_type;
  std::vector<i32> device_ids;
  i32 max_input_count;
  i32 max_frame_width;
  i32 max_frame_height;
};

class EvaluatorFactory {
 public:
  virtual ~EvaluatorFactory(){};

  virtual EvaluatorCapabilities get_capabilities() = 0;

  virtual std::vector<std::string> get_output_names() = 0;

  /* new_evaluator - constructs an evaluator to be used for processing
   *   decoded frames. This function must be thread-safe but evaluators
   *   themself do not need to be.
   */
  virtual Evaluator* new_evaluator(const EvaluatorConfig& config) = 0;
};
}
