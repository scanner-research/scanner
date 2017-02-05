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

#include "scanner/evaluators/image/image_encoder_evaluator.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"

// For image ingest
#include "bitmap-cpp/bitmap.h"
#include "jpegwrapper/JPEGWriter.h"
#include "lodepng/lodepng.h"

#ifdef HAVE_CUDA
#include "scanner/util/cuda.h"
#endif

namespace scanner {

ImageEncoderEvaluator::ImageEncoderEvaluator(const EvaluatorConfig &config,
                                             DeviceType device_type,
                                             ImageEncodingType image_type)
    : device_type_(device_type), device_id_(config.device_ids[0]),
      image_type_(image_type) {}

void ImageEncoderEvaluator::configure(const BatchConfig &config) {
  config_ = config;
  assert(config.formats.size() == 1);
  frame_width_ = config.formats[0].width();
  frame_height_ = config.formats[0].height();
}

void ImageEncoderEvaluator::evaluate(const BatchedColumns &input_columns,
                                     BatchedColumns &output_columns) {
  size_t num_inputs = input_columns[0].rows.size();

  if (image_type_ == ImageEncodingType::RAW) {
    for (i32 i = 0; i < num_inputs; ++i) {
      output_columns[0].rows.push_back(input_columns[0].rows[i]);
    }
  } else {
    std::string ext = image_encoding_type_to_string(image_type_);
    std::vector<u8> tmp_buf;
    for (size_t i = 0; i < num_inputs; ++i) {
      cv::Mat img =
          bytesToImage(input_columns[0].rows[i].buffer, config_.formats[0]);
      bool success = cv::imencode(".jpg", img, tmp_buf);
      LOG_IF(FATAL, !success) << "Failed to encode jpeg";

      size_t out_size = tmp_buf.size();
      u8 *output_buf = new_buffer({device_type_, device_id_}, out_size);
      std::memcpy(output_buf, tmp_buf.data(), out_size);
      output_columns[0].rows.push_back(Row{output_buf, out_size});
    }
  }
}

ImageEncoderEvaluatorFactory::ImageEncoderEvaluatorFactory(
    DeviceType device_type, ImageEncodingType image_type)
    : device_type_(device_type), image_type_(image_type) {}

EvaluatorCapabilities ImageEncoderEvaluatorFactory::get_capabilities() {
  assert(device_type_ == DeviceType::CPU);

  EvaluatorCapabilities caps;
  caps.device_type = device_type_;
  caps.max_devices = 1;
  caps.warmup_size = 0;
  caps.can_overlap = false;
  return caps;
}

std::vector<std::string> ImageEncoderEvaluatorFactory::get_output_columns(
    const std::vector<std::string> &input_columns) {
  return {"image"};
}

Evaluator *
ImageEncoderEvaluatorFactory::new_evaluator(const EvaluatorConfig &config) {
  return new ImageEncoderEvaluator(config, device_type_, image_type_);
}
}
