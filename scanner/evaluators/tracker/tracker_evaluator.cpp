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

#include "struck/Tracker.h"
#include "struck/Config.h"

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
  trackers_.clear();
  LOG(INFO) << "Tracker reset";
}

void TrackerEvaluator::evaluate(
    const std::vector<std::vector<u8 *>> &input_buffers,
    const std::vector<std::vector<size_t>> &input_sizes,
    std::vector<std::vector<u8 *>> &output_buffers,
    std::vector<std::vector<size_t>> &output_sizes) {
  assert(input_buffers.size() >= 2);

  i32 input_count = input_buffers[0].size();

  LOG(INFO) << "Tracker evaluate " << input_count;
  for (int i = 0; i < input_count; ++i) {
    u8 *buf = new u8[10000];
    output_buffers[0].push_back(buf);
    output_sizes[0].push_back(10000);
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

std::vector<std::string> TrackerEvaluatorFactory::get_output_names() {
  return {"image", "before_bboxes", "after_bboxes"};
}

Evaluator *
TrackerEvaluatorFactory::new_evaluator(const EvaluatorConfig &config) {
  return new TrackerEvaluator(config, device_type_, 0, warmup_count_);
}
}
