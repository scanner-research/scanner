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

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

namespace scanner {

class VGGGPUInputTransformer : public CaffeInputTransformer {
public:
  void configure(const DatasetItemMetadata& metadata) override;

  void transform_input(
    char* input_buffer,
    float* net_input,
    int batch_size) override;

private:
  static const int NET_INPUT_WIDTH = 224;
  static const int NET_INPUT_HEIGHT = 224;

  DatasetItemMetadata metadata_;

  cv::Mat input_mat;
  cv::Mat conv_input;
  cv::Mat conv_planar_input;
  cv::Mat float_conv_input;
  cv::Mat normed_input;
};

class VGGGPUInputTransformerFactory : public CaffeInputTransformerFactory {
public:
  CaffeInputTransformer* construct(const EvaluatorConfig& config) override;
};

}
