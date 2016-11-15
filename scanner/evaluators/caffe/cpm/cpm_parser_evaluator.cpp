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

#include "scanner/evaluators/caffe/cpm/cpm_parser_evaluator.h"
#include "scanner/evaluators/serialize.h"

#include "scanner/util/common.h"
#include "scanner/util/util.h"

#include <cassert>
#include <cmath>

namespace scanner {

CPMParserEvaluator::CPMParserEvaluator(const EvaluatorConfig& config,
                                       DeviceType device_type, i32 device_id,
                                       bool forward_input)
    : config_(config),
      device_type_(device_type),
      device_id_(device_id),
      forward_input_(forward_input) {
  if (device_type_ == DeviceType::GPU) {
    LOG(FATAL) << "GPU CPM parser support not implemented yet";
  }
}

void CPMParserEvaluator::configure(const InputFormat& metadata) {
  metadata_ = metadata;

  f32 scale = static_cast<f32>(box_size_) / metadata.height();
  // Calculate width by scaling by box size
  resize_width_ = metadata.width() * scale;
  resize_height_ = metadata.height() * scale;

  width_padding_ = (resize_width_ % 8) ? 8 - (resize_width_ % 8) : 0;
  padded_width_ = resize_width_ + width_padding_;

  net_input_width_ = box_size_;
  net_input_height_ = box_size_;

  feature_width_ = net_input_width_ / cell_size_;
  feature_height_ = net_input_height_ / cell_size_;

  resized_c_ = cv::Mat(net_input_height_, net_input_width_, CV_32FC1);
}

void CPMParserEvaluator::evaluate(
    const std::vector<std::vector<u8*>>& input_buffers,
    const std::vector<std::vector<size_t>>& input_sizes,
    std::vector<std::vector<u8*>>& output_buffers,
    std::vector<std::vector<size_t>>& output_sizes) {
  i32 input_count = (i32)input_buffers[0].size();

  i32 frame_idx = 0;
  i32 feature_idx;
  if (forward_input_) {
    assert(input_buffers.size() >= 2);
    feature_idx = 1;
  } else {
    assert(input_buffers.size() >= 1);
    feature_idx = 0;
  }

  // Get bounding box data from output feature vector and turn it
  // into canonical center x, center y, width, height
  for (i32 b = 0; b < input_count; ++b) {
    assert(input_sizes[feature_idx][b] ==
           feature_width_ * feature_height_ * sizeof(f32));
    cv::Mat input(feature_height_, feature_width_, CV_32FC1,
                  input_buffers[feature_idx][b]);
    cv::resize(input, resized_c_,
               cv::Size(net_input_width_, net_input_height_));
    cv::dilate(resized_c_, max_c_, dilate_kernel_);
    // Remove elements less than threshold
    cv::threshold(max_c_, max_c_, threshold_, 0.0, cv::THRESH_TOZERO);
    // Remove all non maximums
    cv::compare(max_c_, resized_c_, max_c_, cv::CMP_EQ);
    std::vector<cv::Point> maximums;
    // All non-zeros are maximums
    cv::findNonZero(max_c_, maximums);

    std::vector<scanner::Point> centers;
    for (cv::Point p : maximums) {
      scanner::Point pt;
      pt.set_x(p.x);
      pt.set_y(p.y);
      centers.push_back(pt);
      printf("center %d, %d\n", p.x, p.y);
    }
    // Assume size of a bounding box is the same size as all bounding boxes
    size_t size;
    u8* buffer;
    serialize_proto_vector(centers, buffer, size);
    output_buffers[feature_idx].push_back(buffer);
    output_sizes[feature_idx].push_back(size);
  }

  if (forward_input_) {
    for (i32 b = 0; b < input_count; ++b) {
      output_buffers[frame_idx].push_back(input_buffers[frame_idx][b]);
      output_sizes[frame_idx].push_back(input_sizes[frame_idx][b]);
    }
  }
}

CPMParserEvaluatorFactory::CPMParserEvaluatorFactory(DeviceType device_type,
                                                     bool forward_input)
    : device_type_(device_type), forward_input_(forward_input) {}

EvaluatorCapabilities CPMParserEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = device_type_;
  if (device_type_ == DeviceType::GPU) {
    LOG(FATAL) << "GPU CPM parser support not implemented yet";
    caps.max_devices = 1;
  } else {
    caps.max_devices = 1;
  }
  caps.warmup_size = 0;
  return caps;
}

std::vector<std::string> CPMParserEvaluatorFactory::get_output_names() {
  std::vector<std::string> output_names;
  if (forward_input_) {
    output_names.push_back("frame");
  }
  output_names.push_back("centers");
  return output_names;
}

Evaluator* CPMParserEvaluatorFactory::new_evaluator(
    const EvaluatorConfig& config) {
  return new CPMParserEvaluator(config, device_type_, 0, forward_input_);
}
}
