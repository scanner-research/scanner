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

#include "scanner/eval/evaluator.h"
#include "scanner/eval/evaluator_factory.h"
#include "scanner/evaluators/caffe/net_descriptor.h"
#include "scanner/util/opencv.h"

//#define USE_CUDA

#ifdef HAVE_CUDA
#include "scanner/util/cuda.h"
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif

namespace scanner {

class FacenetInputEvaluator : public Evaluator {
 public:
  FacenetInputEvaluator(DeviceType device_type, i32 device_id,
                        const NetDescriptor& descriptor, i32 batch_size);

  void configure(const VideoMetadata& metadata) override;

  void evaluate(const std::vector<std::vector<u8 *>> &input_buffers,
                const std::vector<std::vector<size_t>> &input_sizes,
                std::vector<std::vector<u8 *>> &output_buffers,
                std::vector<std::vector<size_t>> &output_sizes) override;

 private:
  DeviceType device_type_;
  i32 device_id_;
  NetDescriptor descriptor_;
  VideoMetadata metadata_;

  i32 batch_size_;
  i32 net_input_width_;
  i32 net_input_height_;

#ifdef USE_CUDA
  i32 num_cuda_streams_;
  cv::cuda::GpuMat mean_mat_;
  std::vector<cv::cuda::Stream> streams_;
  std::vector<cv::cuda::GpuMat> frame_input_;
  std::vector<cv::cuda::GpuMat> float_input_;
  std::vector<cv::cuda::GpuMat> normalized_input_;
  std::vector<std::vector<cv::cuda::GpuMat>> input_planes_;
  std::vector<std::vector<cv::cuda::GpuMat>> flipped_planes_;
  std::vector<cv::cuda::GpuMat> planar_input_;
#else
  cv::Mat mean_mat_;
  cv::Mat float_input;
  cv::Mat normalized_input;
  cv::Mat flipped_input;
  std::vector<cv::Mat> input_planes;
  cv::Mat planar_input;
#endif
};

class FacenetInputEvaluatorFactory : public EvaluatorFactory {
 public:
  FacenetInputEvaluatorFactory(DeviceType device_type,
                               const NetDescriptor& descriptor,
                               i32 batch_size);

  EvaluatorCapabilities get_capabilities() override;

  std::vector<std::string> get_output_names() override;

  Evaluator* new_evaluator(const EvaluatorConfig& config) override;

private:
  DeviceType device_type_;
  NetDescriptor net_descriptor_;
  i32 batch_size_;
};
}
