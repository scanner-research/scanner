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

#include "scanner/evaluators/image/image_decoder_evaluator.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"

#ifdef HAVE_CUDA
#include "scanner/util/cuda.h"
#endif

namespace scanner {

ImageDecoderEvaluator::ImageDecoderEvaluator(const EvaluatorConfig& config,
                                             DeviceType device_type)
    : device_type_(device_type), device_id_(config.device_ids[0]) {}

void ImageDecoderEvaluator::evaluate(const BatchedColumns& input_columns,
                                     BatchedColumns& output_columns) {
  i32 width = 640;
  i32 height = 480;
  size_t image_size = width * height * 3;
  size_t num_inputs = input_columns.empty() ? 0 : input_columns[0].rows.size();
  u8* output_block = new_block_buffer({device_type_, device_id_},
                                      image_size * num_inputs,
                                      num_inputs);

  for (size_t i = 0; i < num_inputs; ++i) {
    size_t input_size = input_columns[0].rows[i].size;
    cv::Mat input(input_size, 1, CV_8U, input_columns[0].rows[i].buffer);
    u8* output_buf = output_block + i * image_size;
    cv::Mat output(height, width, CV_8UC3, output_buf);
    cv::imdecode(input, cv::IMREAD_COLOR, &output);
    output_columns[0].rows.push_back(Row{output_buf, image_size});
  }
}

ImageDecoderEvaluatorFactory::ImageDecoderEvaluatorFactory(
    DeviceType device_type)
    : device_type_(device_type) {}

EvaluatorCapabilities ImageDecoderEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = device_type_;
  caps.max_devices = 1;
  caps.warmup_size = 0;
  caps.can_overlap = false;
  return caps;
}

std::vector<std::string> ImageDecoderEvaluatorFactory::get_output_columns(
    const std::vector<std::string>& input_columns) {
  return {"frame"};
}

Evaluator* ImageDecoderEvaluatorFactory::new_evaluator(
    const EvaluatorConfig& config) {
  return new ImageDecoderEvaluator(config, device_type_);
}
}
