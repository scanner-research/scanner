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

#include "scanner/evaluators/movie_analysis/face_evaluator.h"

namespace scanner {

FaceEvaluator::FaceEvaluator(EvaluatorConfig) {
  if (!face_detector.load(
          "/opt/stanford/wcrichto/haarcascade_frontalface_alt.xml")) {
    LOG(FATAL) << "Failed to load face cascade";
  }
}

FaceEvaluator::~FaceEvaluator() {}

void FaceEvaluator::configure(const DatasetItemMetadata& metadata) {
  this->metadata = metadata;
}

void FaceEvaluator::evaluate(u8* input_buffer,
                             std::vector<std::vector<u8*>>& output_buffers,
                             std::vector<std::vector<size_t>>& output_sizes,
                             i32 batch_size) {
  for (i32 i = 0; i < batch_size; ++i) {
    cv::Mat img = bytesToImage(input_buffer, i, metadata);
    std::vector<cv::Rect> faces;
    face_detector.detectMultiScale(img, faces, 1.1, 2, cv::CASCADE_SCALE_IMAGE,
                                   cv::Size(30, 30));

    i32 size = faces.size() * sizeof(i32) * 4;
    i32* output_buffer = (i32*)(new u8[size]);
    for (i32 j = 0; j < faces.size(); ++j) {
      cv::Rect& face = faces[j];
      i32* offset = output_buffer + j * 4;
      *(offset + 0) = face.x;
      *(offset + 1) = face.y;
      *(offset + 2) = face.width;
      *(offset + 3) = face.height;
    }

    output_buffers[0].push_back((u8*)output_buffer);
    output_sizes[0].push_back(size);
  }
}

FaceEvaluatorConstructor::FaceEvaluatorConstructor() {}

FaceEvaluatorConstructor::~FaceEvaluatorConstructor() {}

i32 FaceEvaluatorConstructor::get_number_of_devices() { return 1; }

DeviceType FaceEvaluatorConstructor::get_input_buffer_type() {
  return DeviceType::CPU;
}

DeviceType FaceEvaluatorConstructor::get_output_buffer_type() {
  return DeviceType::CPU;
}

i32 FaceEvaluatorConstructor::get_number_of_outputs() { return 1; }

std::vector<std::string> FaceEvaluatorConstructor::get_output_names() {
  return {"face"};
}

u8* FaceEvaluatorConstructor::new_input_buffer(const EvaluatorConfig& config) {
  return new u8[config.max_batch_size * config.max_frame_width *
                config.max_frame_height * 3 * sizeof(u8)];
}

void FaceEvaluatorConstructor::delete_input_buffer(
    const EvaluatorConfig& config, u8* buffer) {
  delete[] buffer;
}

void FaceEvaluatorConstructor::delete_output_buffer(
    const EvaluatorConfig& config, u8* buffer) {
  delete[] buffer;
}

Evaluator* FaceEvaluatorConstructor::new_evaluator(
    const EvaluatorConfig& config) {
  return new FaceEvaluator(config);
}
}
