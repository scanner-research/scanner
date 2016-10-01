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

#include "scanner/evaluators/image_processing/blur_evaluator.h"

#include <cmath>

namespace scanner {

BlurEvaluator::BlurEvaluator(EvaluatorConfig config, i32 kernel_size, f64 sigma)
    : kernel_size_(kernel_size),
      filter_left_(std::ceil(kernel_size / 2.0) - 1),
      filter_right_(kernel_size / 2),
      sigma_(sigma) {}

void BlurEvaluator::configure(const DatasetItemMetadata& metadata) {
  metadata_ = metadata;
}

void BlurEvaluator::evaluate(
    const std::vector<std::vector<u8 *>> &input_buffers,
    const std::vector<std::vector<size_t>> &input_sizes,
    std::vector<std::vector<u8 *>> &output_buffers,
    std::vector<std::vector<size_t>> &output_sizes) {

  i32 input_count = (i32)input_buffers[0].size();
  // HACK(apoms): we can only do this because we know the decoder stores all
  //   frames contiguously in memory right now
  u8 *input_buffer = input_buffers[0][0];

  i32 width = metadata_.width;
  i32 height = metadata_.height;
  size_t frame_size = width * height * 3 * sizeof(u8);

  for (i32 i = 0; i < input_count; ++i) {
    u8* output_buffer = new u8[frame_size];

    u8* frame_buffer = (input_buffer + frame_size * i);
    u8* blurred_buffer = (output_buffer);
    for (i32 y = filter_left_; y < height - filter_right_; ++y) {
      for (i32 x = filter_left_; x < width - filter_right_; ++x) {
        for (i32 c = 0; c < 3; ++c) {
          u32 value = 0;
          for (i32 ry = -filter_left_; ry < filter_right_ + 1; ++ry) {
            for (i32 rx = -filter_left_; rx < filter_right_ + 1; ++rx) {
              value += frame_buffer[(y + ry) * width * 3 + (x + rx) * 3 + c];
            }
          }
          blurred_buffer[y * width * 3 + x * 3 + c] =
              value / ((filter_right_ + filter_left_ + 1) *
                       (filter_right_ + filter_left_ + 1));
        }
      }
    }
    output_sizes[0].push_back(frame_size);
    output_buffers[0].push_back(output_buffer);
  }
}

BlurEvaluatorFactory::BlurEvaluatorFactory(i32 kernel_size, f64 sigma)
    : kernel_size_(kernel_size), sigma_(sigma) {}

EvaluatorCapabilities BlurEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = DeviceType::CPU;
  caps.max_devices = 1;
  caps.warmup_size = 0;
  return caps;
}

std::vector<std::string> BlurEvaluatorFactory::get_output_names() {
  return {"image"};
}

Evaluator* BlurEvaluatorFactory::new_evaluator(const EvaluatorConfig& config) {
  return new BlurEvaluator(config, kernel_size_, sigma_);
}
}
