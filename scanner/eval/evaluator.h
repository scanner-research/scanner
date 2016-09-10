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

enum class BufferType {
  GPU,
  CPU,
};

struct EvaluatorConfig {
public:
  int max_batch_size;
  int num_staging_buffers;
};

class Evaluator {
public:
  virtual ~Evaluator() {};

  virtual int get_max_batch_size() = 0;

  virtual int get_num_staging_buffers() = 0;

  virtual char* get_staging_buffer(int index) = 0;

  virtual void evaluate(
    int staging_buffer_index,
    char* output,
    int batch_size) = 0;
};

class EvaluatorConstructor {
  virtual ~EvaluatorConstructor() {};

  virtual BufferType get_input_buffer_type() = 0;

  virtual BufferType get_output_buffer_type() = 0;

  virtual int get_number_of_devices() = 0;

  virtual Evaluator* new_evaluator(
    int device_id,
    int max_batch_size,
    int num_staging_buffers,
    size_t staging_buffer_size) = 0;
};



// allocate buffers
// setup
// set batch size
// consume input
// produce output
// teardown
// deallocate buffers

}
