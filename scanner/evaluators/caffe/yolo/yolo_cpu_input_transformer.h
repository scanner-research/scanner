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

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace scanner {

class YoloCPUInputTransformer : public CaffeInputTransformer {
 public:
  YoloCPUInputTransformer(const NetDescriptor& descriptor);

  void configure(const DatasetItemMetadata& metadata,
                 caffe::Net<float>* net) override;

  void transform_input(u8* input_buffer, f32* net_input,
                       i32 batch_size) override;

 private:
  static const i32 NET_INPUT_WIDTH = 448;
  static const i32 NET_INPUT_HEIGHT = 448;

  NetDescriptor descriptor_;
  DatasetItemMetadata metadata_;

  cv::Mat mean_mat;

  cv::Mat resized_input;
  cv::Mat bgr_input;
  std::vector<cv::Mat> input_planes;
  cv::Mat planar_input;
  cv::Mat float_input;
  cv::Mat normalized_input;
};

class YoloCPUInputTransformerFactory : public CaffeInputTransformerFactory {
 public:
  CaffeInputTransformer* construct(const EvaluatorConfig& config,
                                   const NetDescriptor& descriptor) override;
};
}
