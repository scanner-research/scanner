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

#include "scanner/evaluators/caffe/cpm/cpm_person_parser_evaluator.h"
#include "scanner/evaluators/serialize.h"

#include "scanner/util/common.h"
#include "scanner/util/util.h"

#ifdef HAVE_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#endif

#include <cassert>
#include <cmath>

namespace scanner {

CPMPersonParserEvaluator::CPMPersonParserEvaluator(
    const EvaluatorConfig &config, DeviceType device_type, i32 device_id)
    : device_type_(device_type), device_id_(device_id) {
  if (device_type_ == DeviceType::GPU) {
    LOG(FATAL) << "GPU CPM person parser support not implemented yet";
  }
}

void CPMPersonParserEvaluator::configure(const BatchConfig &config) {
  config_ = config;
  assert(config.formats.size() == 1);
  metadata_ = config.formats[0];

  f32 scale = static_cast<f32>(box_size_) / metadata_.height();
  // Calculate width by scaling by box size
  resize_width_ = metadata_.width() * scale;
  resize_height_ = metadata_.height() * scale;

  width_padding_ = (resize_width_ % 8) ? 8 - (resize_width_ % 8) : 0;

  net_input_width_ = resize_width_ + width_padding_;
  net_input_height_ = resize_height_;

  feature_width_ = net_input_width_ / cell_size_;
  feature_height_ = net_input_height_ / cell_size_;

  dilate_kernel_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  resized_c_ = cv::Mat(net_input_height_, net_input_width_, CV_32FC1);
  max_c_ = cv::Mat(net_input_height_, net_input_width_, CV_32FC1);
}

void CPMPersonParserEvaluator::evaluate(const BatchedColumns &input_columns,
                                        BatchedColumns &output_columns) {
  i32 input_count = (i32)input_columns[0].rows.size();

  i32 frame_idx = 0;
  i32 feature_idx;
  assert(input_columns.size() >= 2);
  feature_idx = 1;

  // Get bounding box data from output feature vector and turn it
  // into canonical center x, center y, width, height
  for (i32 b = 0; b < input_count; ++b) {
    assert(input_columns[feature_idx].rows[b].size ==
           feature_width_ * feature_height_ * sizeof(f32));
    cv::Mat input(feature_height_, feature_width_, CV_32FC1,
                  input_columns[feature_idx].rows[b].buffer);
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
    }
    // Assume size of a bounding box is the same size as all bounding boxes
    size_t size;
    u8 *buffer;
    serialize_proto_vector(centers, buffer, size);
    output_columns[feature_idx].rows.push_back(Row{buffer, size});
  }

  for (i32 b = 0; b < input_count; ++b) {
    output_columns[frame_idx].rows.push_back(input_columns[frame_idx].rows[b]);
  }
}

CPMPersonParserEvaluatorFactory::CPMPersonParserEvaluatorFactory(
    DeviceType device_type)
    : device_type_(device_type) {}

EvaluatorCapabilities CPMPersonParserEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = device_type_;
  if (device_type_ == DeviceType::GPU) {
    LOG(FATAL) << "GPU CPM person parser support not implemented yet";
    caps.max_devices = 1;
  } else {
    caps.max_devices = 1;
  }
  caps.warmup_size = 0;
  return caps;
}

std::vector<std::string> CPMPersonParserEvaluatorFactory::get_output_columns(
    const std::vector<std::string> &input_columns) {
  std::vector<std::string> output_names;
  output_names.push_back("frame");
  output_names.push_back("centers");
  return output_names;
}

Evaluator *
CPMPersonParserEvaluatorFactory::new_evaluator(const EvaluatorConfig &config) {
  return new CPMPersonParserEvaluator(config, device_type_,
                                      config.device_ids[0]);
}
}
