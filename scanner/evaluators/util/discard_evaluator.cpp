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

#include "scanner/evaluators/util/discard_evaluator.h"

#include "scanner/util/common.h"
#include "scanner/util/memory.h"

namespace scanner {

DiscardEvaluator::DiscardEvaluator(const EvaluatorConfig& config,
                                   DeviceType device_type, i32 device_id)
    : config_(config), device_type_(device_type), device_id_(device_id) {}

void DiscardEvaluator::configure(const InputFormat& metadata) {
  printf("configure %d, %d\n", metadata.width(), metadata.height());
}

void DiscardEvaluator::evaluate(
    const std::vector<std::vector<u8*>>& input_buffers,
    const std::vector<std::vector<size_t>>& input_sizes,
    std::vector<std::vector<u8*>>& output_buffers,
    std::vector<std::vector<size_t>>& output_sizes) {
  i32 input_count = static_cast<i32>(input_buffers[0].size());
  for (i32 i = 0; i < input_count; ++i) {
    printf("input size %lu\n", input_sizes[0][i]);
    output_buffers[0].push_back(new_buffer(device_type_, device_id_, 1));
    output_sizes[0].push_back(1);
  }
}

DiscardEvaluatorFactory::DiscardEvaluatorFactory(DeviceType device_type)
    : device_type_(device_type) {}

EvaluatorCapabilities DiscardEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = device_type_;
  caps.max_devices = 1;
  caps.warmup_size = 0;
  return caps;
}

std::vector<std::string> DiscardEvaluatorFactory::get_output_names() {
  return {"discard"};
}

Evaluator* DiscardEvaluatorFactory::new_evaluator(
    const EvaluatorConfig& config) {
  return new DiscardEvaluator(config, device_type_, config.device_ids[0]);
}
}
