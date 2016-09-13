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

#include "scanner/eval/image_processing/blur_evaluator.h"

namespace scanner {

BlurEvaluator::BlurEvaluator(EvaluatorConfig config, double sigma)
  : sigma_(sigma)
{
}

BlurEvaluator::~BlurEvaluator() {
}

void BlurEvaluator::evaluate(
  char* input_buffer,
  std::vector<char*> output_buffers,
  int batch_size)
{
}

BlurEvaluatorConstructor::BlurEvaluatorConstructor(double sigma)
  : sigma_(sigma)
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
  return new BlurEvaluator(config, sigma_);
}

}
