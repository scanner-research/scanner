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

#include "scanner/evaluators/tracker/tracker_evaluator.h"

#include "scanner/util/common.h"
#include "scanner/util/util.h"

namespace scanner {

TrackerEvaluator::TrackerEvaluator(const EvaluatorConfig& config,
                                   DeviceType device_type,
                                   i32 device_id,
                                   i32 warmup_count)
    : config_(config),
      device_type_(device_type),
      device_id_(device_id),
      warmup_count_(warmup_count)
{
  if (device_type_ == DeviceType::GPU) {
    LOG(FATAL) << "GPU tracker support not implemented yet";
  }
}

void TrackerEvaluator::configure(const DatasetItemMetadata& metadata) {
  metadata_ = metadata;
  LOG(INFO) << "Tracker configure";
}

void TrackerEvaluator::reset() {
  LOG(INFO) << "Tracker reset";
}

void TrackerEvaluator::warmup(i32 input_count, u8* input_buffer) {
  LOG(INFO) << "Tracker warmup " << input_count;
}

void TrackerEvaluator::evaluate(i32 input_count, u8* input_buffer,
                              std::vector<std::vector<u8*>>& output_buffers,
                              std::vector<std::vector<size_t>>& output_sizes) {
  LOG(INFO) << "Tracker evaluate " << input_count;
  for (int i = 0; i < input_count; ++i) {
    u8 *buf = new u8[1];
    output_buffers[0].push_back(buf);
    output_sizes[0].push_back(1);
  }
}

TrackerEvaluatorFactory::TrackerEvaluatorFactory(DeviceType device_type,
                                                 i32 warmup_count)
    : device_type_(device_type), warmup_count_(warmup_count) {
  if (device_type_ == DeviceType::GPU) {
    LOG(FATAL) << "GPU tracker support not implemented yet";
  }
}

EvaluatorCapabilities TrackerEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = device_type_;
  caps.max_devices = 1;
  caps.warmup_size = warmup_count_;
  return caps;
}

i32 TrackerEvaluatorFactory::get_number_of_outputs() {
  return 1;
}

std::vector<std::string> TrackerEvaluatorFactory::get_output_names() {
  return {"track"};
}

Evaluator *
TrackerEvaluatorFactory::new_evaluator(const EvaluatorConfig &config) {
  return new TrackerEvaluator(config, device_type_, 0, warmup_count_);
}
}
