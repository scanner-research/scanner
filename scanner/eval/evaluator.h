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

namespace scanner {

enum class DeviceType {
  GPU,
  CPU,
};

class Evaluator {
public:
  virtual ~Evaluator() {};

  virtual void evaluate(
    char* input_buffer,
    std::vector<char*> output_buffers,
    int batch_size) = 0;
};

struct EvaluatorConfig {
  int device_id;
  int max_batch_size;
  size_t staging_buffer_size;
  int frame_width;
  int frame_height;
};

class EvaluatorConstructor {
  virtual ~EvaluatorConstructor() {};

  virtual int get_number_of_devices() = 0;

  virtual DeviceType get_input_buffer_type() = 0;

  virtual DeviceType get_output_buffer_type() = 0;

  virtual int get_number_of_outputs() = 0;

  virtual std::vector<std::string> get_output_names() = 0;

  virtual std::vector<size_t> get_output_element_sizes() = 0;

  virtual char* new_input_buffer(const EvaluatorConfig& config) = 0;

  virtual void delete_input_buffer(
    const EvaluatorConfig& config,
    char* buffer) = 0;

  virtual std::vector<char*> new_output_buffers(
    const EvaluatorConfig& config,
    int num_inputs) = 0;

  virtual void delete_output_buffers(
    const EvaluatorConfig& config,
    std::vector<char*> buffers) = 0;

  /* new_evaluator - constructs an evaluator to be used for processing
   *   decoded frames. Must be thread-safe.
   */
  virtual Evaluator* new_evaluator(const EvaluatorConfig& config) = 0;
};

// allocate buffers
// setup
// set batch size
// consume input
// produce output
// teardown
// deallocate buffers

}
