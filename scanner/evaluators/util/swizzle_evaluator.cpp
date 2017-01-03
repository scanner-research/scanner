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

#include "scanner/evaluators/util/swizzle_evaluator.h"

#include "scanner/util/common.h"
#include "scanner/util/memory.h"
#include "scanner/util/util.h"

#ifdef HAVE_CUDA
#include "scanner/util/cuda.h"
#endif

#include <cassert>

namespace scanner {

SwizzleEvaluator::SwizzleEvaluator(const EvaluatorConfig& config,
                                   DeviceType device_type, i32 device_id,
                                   const std::vector<i32>& output_to_input_idx)
    : device_type_(device_type),
      device_id_(device_id),
      output_to_input_idx_(output_to_input_idx) {}

void SwizzleEvaluator::configure(const BatchConfig& config) {
  config_ = config;
}

void SwizzleEvaluator::evaluate(const BatchedColumns& input_columns,
                                BatchedColumns& output_columns) {
  // i32 input_count = static_cast<i32>(input_buffers[0].size());
  size_t num_outputs = output_to_input_idx_.size();
  for (size_t i = 0; i < num_outputs; ++i) {
    i32 input_idx = output_to_input_idx_[i];
    assert(input_idx < input_columns.size());
    for (i32 b = 0; b < (i32)input_columns[input_idx].rows.size(); ++b) {
      output_columns[i].rows.push_back(input_columns[input_idx].rows[b]);
    }
  }
}

SwizzleEvaluatorFactory::SwizzleEvaluatorFactory(
    DeviceType device_type, const std::vector<i32>& output_to_input_idx,
    const std::vector<std::string>& output_names)
    : device_type_(device_type),
      output_to_input_idx_(output_to_input_idx),
      output_names_(output_names) {
  assert(output_names.size() == output_to_input_idx.size());
}

EvaluatorCapabilities SwizzleEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = device_type_;
  caps.max_devices = 1;
  caps.warmup_size = 0;
  return caps;
}

std::vector<std::string> SwizzleEvaluatorFactory::get_output_columns(
    const std::vector<std::string>& input_columns) {
  return output_names_;
}

Evaluator* SwizzleEvaluatorFactory::new_evaluator(
    const EvaluatorConfig& config) {
  return new SwizzleEvaluator(config, device_type_, 0, output_to_input_idx_);
}
}
