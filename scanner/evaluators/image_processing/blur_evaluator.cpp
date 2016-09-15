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

BlurEvaluator::BlurEvaluator(
  EvaluatorConfig config,
  int kernel_size,
  double sigma)
  : kernel_size_(kernel_size),
    filter_left_(std::ceil(kernel_size / 2.0) - 1),
    filter_right_(kernel_size / 2),
    sigma_(sigma)
{
}

BlurEvaluator::~BlurEvaluator() {
}


void BlurEvaluator::configure(const DatasetItemMetadata& metadata) {
  metadata_ = metadata;
}

void BlurEvaluator::evaluate(
  char* input_buffer,
  std::vector<char*> output_buffers,
  int batch_size)
{
  int width = metadata_.width;
  int height = metadata_.height;

  char* output_buffer = output_buffers[0];
  int64_t frame_size = width * height * 3 * sizeof(char);
  for (int i = 0; i < batch_size; ++i) {
    uint8_t* frame_buffer = (uint8_t*)(input_buffer + frame_size * i);
    uint8_t* blurred_buffer = (uint8_t*)(output_buffer + frame_size * i);
    for (int y = filter_left_; y < height - filter_right_; ++y) {
      for (int x = filter_left_; x < width - filter_right_; ++x) {
        for (int c = 0; c < 3; ++c) {
          uint32_t value = 0;
          for (int ry = -filter_left_; ry < filter_right_ + 1; ++ry) {
            for (int rx = -filter_left_; rx < filter_right_ + 1; ++rx) {
              value += frame_buffer[(y + ry) * width * 3 +
                                    (x + rx) * 3 +
                                    c];
            }
          }
          blurred_buffer[y * width * 3 + x * 3 + c] =
            value / ((filter_right_ + filter_left_ + 1) *
                     (filter_right_ + filter_left_ + 1));
        }
      }
    }
  }
}

BlurEvaluatorConstructor::BlurEvaluatorConstructor(
  int kernel_size,
  double sigma)
  : kernel_size_(kernel_size),
    sigma_(sigma)
{
}

BlurEvaluatorConstructor::~BlurEvaluatorConstructor() {
}

int BlurEvaluatorConstructor::get_number_of_devices() {
  return 1;
}

DeviceType BlurEvaluatorConstructor::get_input_buffer_type() {
  return DeviceType::CPU;
}

DeviceType BlurEvaluatorConstructor::get_output_buffer_type() {
  return DeviceType::CPU;
}

int BlurEvaluatorConstructor::get_number_of_outputs() {
  return 1;
}

std::vector<std::string> BlurEvaluatorConstructor::get_output_names() {
  return {"image"};
}

std::vector<size_t> BlurEvaluatorConstructor::get_output_element_sizes(
  const EvaluatorConfig& config)
{
  return {config.max_frame_width * config.max_frame_height * 3 * sizeof(char)};
}

char*
BlurEvaluatorConstructor::new_input_buffer(const EvaluatorConfig& config) {
  return new char[
    config.max_batch_size *
    config.max_frame_width *
    config.max_frame_height *
    3 *
    sizeof(char)];
}

void BlurEvaluatorConstructor::delete_input_buffer(
  const EvaluatorConfig& config,
  char* buffer)
{
  delete[] buffer;
}

std::vector<char*> BlurEvaluatorConstructor::new_output_buffers(
  const EvaluatorConfig& config,
  int num_inputs)
{
  return {new char[
      num_inputs *
      config.max_frame_width *
      config.max_frame_height *
      3 *
      sizeof(char)]};
}

void BlurEvaluatorConstructor::delete_output_buffers(
  const EvaluatorConfig& config,
  std::vector<char*> buffers)
{
  delete[] buffers[0];
}

Evaluator*
BlurEvaluatorConstructor::new_evaluator(const EvaluatorConfig& config) {
  return new BlurEvaluator(config, kernel_size_, sigma_);
}

}
