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

#include "scanner/evaluators/caffe/caffe_input_transformer.h"
#include "scanner/evaluators/caffe/caffe_input_transformer_factory.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

#include "scanner/util/cuda.h"

namespace scanner {

class FacenetGPUInputTransformer : public CaffeInputTransformer {
 public:
  FacenetGPUInputTransformer(const NetDescriptor& descriptor);

  void configure(const VideoMetadata& metadata,
                 caffe::Net<float>* net) override;

  void transform_input(i32 input_count, u8* input_buffer,
                       f32* net_input) override;

 private:
  void initialize();

  NetDescriptor descriptor_;
  VideoMetadata metadata_;
  i32 num_cuda_streams_;

  i32 net_input_width_;
  i32 net_input_height_;

  cv::cuda::GpuMat mean_mat_;

  std::vector<cv::cuda::Stream> streams_;
  std::vector<cv::cuda::GpuMat> frame_input_;
  std::vector<cv::cuda::GpuMat> float_input_;
  std::vector<cv::cuda::GpuMat> normalized_input_;
  std::vector<std::vector<cv::cuda::GpuMat>> input_planes_;
  std::vector<std::vector<cv::cuda::GpuMat>> flipped_planes_;
  std::vector<cv::cuda::GpuMat> planar_input_;
};

class FacenetGPUInputTransformerFactory : public CaffeInputTransformerFactory {
 public:
  CaffeInputTransformer* construct(const EvaluatorConfig& config,
                                   const NetDescriptor& descriptor) override;
};
}
