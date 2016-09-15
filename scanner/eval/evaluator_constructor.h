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

struct EvaluatorConfig {
  i32 device_id;
  i32 max_batch_size;
  size_t staging_buffer_size;
  i32 max_frame_width;
  i32 max_frame_height;
};

class EvaluatorConstructor {
public:
  virtual ~EvaluatorConstructor() {};

  virtual i32 get_number_of_devices() = 0;

  virtual DeviceType get_input_buffer_type() = 0;

  virtual DeviceType get_output_buffer_type() = 0;

  virtual i32 get_number_of_outputs() = 0;

  virtual std::vector<std::string> get_output_names() = 0;

  virtual u8* new_input_buffer(const EvaluatorConfig& config) = 0;

  virtual void delete_input_buffer(
    const EvaluatorConfig& config,
    u8* buffer) = 0;

  virtual void delete_output_buffer(
    const EvaluatorConfig& config,
    u8* buffer) = 0;

  /* new_evaluator - constructs an evaluator to be used for processing
   *   decoded frames. Must be thread-safe.
   */
  virtual Evaluator* new_evaluator(const EvaluatorConfig& config) = 0;
};

}
