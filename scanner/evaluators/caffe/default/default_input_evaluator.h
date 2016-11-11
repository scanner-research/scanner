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

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"

#ifdef HAVE_CUDA
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include "scanner/util/cuda.h"
#endif

namespace scanner {

using InputLayerBuilder =
    std::function<void(u8*&, size_t&, const VideoMetadata&)>;

class DefaultInputEvaluator : public Evaluator {
 public:
  DefaultInputEvaluator(DeviceType device_type, i32 device_id,
                        const NetDescriptor& descriptor, i32 batch_size,
                        std::vector<InputLayerBuilder> input_layer_builders);

  void configure(const VideoMetadata& metadata) override;

  void evaluate(const std::vector<std::vector<u8*>>& input_buffers,
                const std::vector<std::vector<size_t>>& input_sizes,
                std::vector<std::vector<u8*>>& output_buffers,
                std::vector<std::vector<size_t>>& output_sizes) override;

  i32 net_input_width_;
  i32 net_input_height_;

 private:
  DeviceType device_type_;
  i32 device_id_;
  NetDescriptor descriptor_;
  VideoMetadata metadata_;
  i32 batch_size_;

#ifdef HAVE_CUDA
  i32 num_cuda_streams_;
  cv::cuda::GpuMat mean_mat_g_;
  std::vector<cv::cuda::Stream> streams_;
  std::vector<cv::cuda::GpuMat> frame_input_g_;
  std::vector<cv::cuda::GpuMat> float_input_g_;
  std::vector<cv::cuda::GpuMat> meanshifted_input_g_;
  std::vector<cv::cuda::GpuMat> normalized_input_g_;
  std::vector<std::vector<cv::cuda::GpuMat>> input_planes_g_;
  std::vector<std::vector<cv::cuda::GpuMat>> flipped_planes_g_;
  std::vector<cv::cuda::GpuMat> planar_input_g_;
#endif

  cv::Mat resized_input_c_;
  std::vector<cv::Mat> input_mats_c_;
  std::unique_ptr<caffe::DataTransformer<f32>> transformer_;
  caffe::Blob<f32> output_blob_;
  std::vector<InputLayerBuilder> input_layer_builders_;
};

class DefaultInputEvaluatorFactory : public EvaluatorFactory {
 public:
  DefaultInputEvaluatorFactory(
      DeviceType device_type, const NetDescriptor& descriptor, i32 batch_size,
      std::vector<InputLayerBuilder> input_layer_builders = {});

  EvaluatorCapabilities get_capabilities() override;

  std::vector<std::string> get_output_names() override;

  Evaluator* new_evaluator(const EvaluatorConfig& config) override;

 private:
  DeviceType device_type_;
  NetDescriptor net_descriptor_;
  i32 batch_size_;
  std::vector<InputLayerBuilder> input_layer_builders_;
};
}
