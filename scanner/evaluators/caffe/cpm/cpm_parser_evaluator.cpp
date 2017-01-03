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
                                       DeviceType device_type, i32 device_id)
    : device_type_(device_type),
      device_id_(device_id)
#ifdef HAVE_CUDA
      ,
      num_cuda_streams_(32),
      streams_(num_cuda_streams_)
#endif
{
}

void CPMParserEvaluator::configure(const BatchConfig& config) {
  config_ = config;
  assert(config.formats.size() == 1);
  metadata_ = config.formats[0];

  f32 scale = static_cast<f32>(box_size_) / metadata_.height();
  // Calculate width by scaling by box size
  resize_width_ = metadata_.width() * scale;
  resize_height_ = metadata_.height() * scale;

  width_padding_ = (resize_width_ % 8) ? 8 - (resize_width_ % 8) : 0;
  padded_width_ = resize_width_ + width_padding_;

  net_input_width_ = box_size_;
  net_input_height_ = box_size_;

  feature_width_ = net_input_width_ / cell_size_;
  feature_height_ = net_input_height_ / cell_size_;
  feature_channels_ = 15;

  if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
    cv::cuda::setDevice(device_id_);
    cudaSetDevice(device_id_);

    streams_.resize(0);
    streams_.resize(num_cuda_streams_);

    resized_g_.clear();
    for (size_t i = 0; i < num_cuda_streams_; ++i) {
      frame_input_g_.push_back(
          cv::cuda::GpuMat(feature_height_, feature_width_, CV_32FC1));
      resized_g_.push_back(
          cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_32FC1));
    }
#else
    LOG(FATAL) << "Not built with CUDA support.";
#endif
  } else {
    resized_c_ = cv::Mat(net_input_height_, net_input_width_, CV_32FC1);
  }
}

void CPMParserEvaluator::evaluate(const BatchedColumns& input_columns,
                                  BatchedColumns& output_columns) {
  i32 frame_idx = 0;
  i32 feature_idx;
  assert(input_columns.size() >= 2);
  feature_idx = 1;

  i32 input_count = (i32)input_columns[feature_idx].rows.size();

  // Get bounding box data from output feature vector and turn it
  // into canonical center x, center y, width, height
  if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
    cv::cuda::setDevice(device_id_);
    cudaSetDevice(device_id_);

    i32* min_max_locs =
        (i32*)malloc(input_count * feature_channels_ * 2 * sizeof(int));
    f32* min_max_values =
        (f32*)malloc(input_count * feature_channels_ * 2 * sizeof(float));

    for (i32 b = 0; b < input_count; ++b) {
      assert(input_columns[feature_idx].rows[b].size ==
             feature_width_ * feature_height_ * feature_channels_ *
                 sizeof(f32));

      for (i32 i = 0; i < feature_channels_; ++i) {
        i32 offset = b * feature_channels_ + i;
        i32 sid = offset % num_cuda_streams_;
        cv::cuda::Stream& s = streams_[sid];

        frame_input_g_[sid] = cv::cuda::GpuMat(
            feature_height_, feature_width_, CV_32FC1,
            input_columns[feature_idx].rows[b].buffer +
                i * feature_width_ * feature_height_ * sizeof(f32));

        cv::cuda::resize(frame_input_g_[sid], resized_g_[sid],
                         cv::Size(net_input_width_, net_input_height_), 0, 0,
                         cv::INTER_NEAREST, s);

        cv::Mat locs(2, 1, CV_32SC1, min_max_locs + offset * 2);
        cv::Mat vals(2, 1, CV_32FC1, min_max_values + offset * 2);
        cv::cuda::findMinMaxLoc(resized_g_[sid], vals, locs, cv::Mat(), s);
      }
    }
    for (cv::cuda::Stream& s : streams_) {
      s.waitForCompletion();
    }

    for (i32 b = 0; b < input_count; ++b) {
      i32 sid = b % num_cuda_streams_;
      std::vector<scanner::Point> pts;
      for (i32 i = 0; i < feature_channels_; ++i) {
        i32 offset = b * feature_channels_ + i;

        i32 max_loc = min_max_locs[2 * offset + 1];
        f32 max_value = min_max_values[2 * offset + 1];

        scanner::Point pt;
        pt.set_x(max_loc % net_input_width_);
        pt.set_y(max_loc / net_input_width_);
        pt.set_score(max_value);
        pts.push_back(pt);
      }

      size_t size;
      u8* buffer;
      serialize_proto_vector(pts, buffer, size);

      // cv::cuda::Stream& s = streams_[sid];
      // cudaStream_t cuda_s = cv::cuda::StreamAccessor::getStream(cv_stream);

      u8* gpu_buffer;
      CU_CHECK(cudaMalloc((void**)&gpu_buffer, size));
      cudaMemcpy(gpu_buffer, buffer, size, cudaMemcpyDefault);
      delete[] buffer;
      output_columns[feature_idx].rows.push_back(Row{gpu_buffer, size});
    }
    free(min_max_locs);
    free(min_max_values);
#endif
  } else {
    for (i32 b = 0; b < input_count; ++b) {
      assert(input_columns[feature_idx].rows[b].size ==
             feature_width_ * feature_height_ * feature_channels_ *
                 sizeof(f32));

      std::vector<scanner::Point> pts;
      for (i32 i = 0; i < feature_channels_; ++i) {
        cv::Mat input(feature_height_, feature_width_, CV_32FC1,
                      input_columns[feature_idx].rows[b].buffer +
                          i * feature_width_ * feature_height_ * sizeof(f32));
        cv::resize(input, resized_c_,
                   cv::Size(net_input_width_, net_input_height_));
        double max_value;
        cv::Point max_location;
        cv::minMaxLoc(resized_c_, NULL, &max_value, NULL, &max_location);

        scanner::Point pt;
        pt.set_x(max_location.x);
        pt.set_y(max_location.y);
        pt.set_score(max_value);
        pts.push_back(pt);
      }

      size_t size;
      u8* buffer;
      serialize_proto_vector(pts, buffer, size);
      output_columns[feature_idx].rows.push_back(Row{buffer, size});
    }
  }

  i32 num_frames = static_cast<i32>(output_columns[frame_idx].rows.size());
  for (i32 b = 0; b < num_frames; ++b) {
    output_columns[frame_idx].rows.push_back(input_columns[frame_idx].rows[b]);
  }
}

CPMParserEvaluatorFactory::CPMParserEvaluatorFactory(DeviceType device_type)
    : device_type_(device_type) {}

EvaluatorCapabilities CPMParserEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = device_type_;
  if (device_type_ == DeviceType::GPU) {
    caps.max_devices = 1;
  } else {
    caps.max_devices = 1;
  }
  caps.warmup_size = 0;
  return caps;
}

std::vector<std::string> CPMParserEvaluatorFactory::get_output_columns(
    const std::vector<std::string>& input_columns) {
  std::vector<std::string> output_names;
  output_names.push_back("frame");
  output_names.push_back("centers");
  return output_names;
}

Evaluator* CPMParserEvaluatorFactory::new_evaluator(
    const EvaluatorConfig& config) {
  return new CPMParserEvaluator(config, device_type_, config.device_ids[0]);
}
}
