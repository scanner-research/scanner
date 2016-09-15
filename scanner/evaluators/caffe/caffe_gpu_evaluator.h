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
#include "scanner/eval/evaluator_constructor.h"
#include "scanner/eval/caffe/net_descriptor.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

namespace scanner {

class CaffeGPUEvaluator : public Evaluator {
public:
  CaffeGPUEvaluator(
    EvaluatorConfig config,
    NetDescriptor descriptor,
    CaffeInputTransformer* transformer,
    int device_id);

  virtual ~CaffeGPUEvaluator();

  virtual void configure(const DatasetItemMetadata& metadata) override;

  virtual void evaluate(
    char* input_buffer,
    std::vector<char*> output_buffers,
    int batch_size) override;

protected:
  NetDescriptor descriptor_;
  CaffeInputTransformer* transformer_;
  int device_id_;
  std::unique_ptr<caffe::Net<float>> net_;
  cv::cuda::GpuMat mean_mat_; // mean image for input normalization

  std::vector<size_t> output_sizes_;

  std::vector<cv::cuda::Stream> cv_streams;
  std::vector<cv::cuda::GpuMat> input_mats;
  std::vector<cv::cuda::GpuMat> rgba_mat;
  std::vector<cv::cuda::GpuMat> rgb_mat;
  std::vector<cv::cuda::GpuMat> conv_input;
  std::vector<cv::cuda::GpuMat> conv_planar_input;
  std::vector<cv::cuda::GpuMat> float_conv_input;
  std::vector<cv::cuda::GpuMat> normed_input;
  std::vector<cv::cuda::GpuMat> scaled_input;
};

class CaffeGPUEvaluatorConstructor : public EvaluatorConstructor {
public:
  CaffeGPUEvaluatorConstructor(
    NetDescriptor net_descriptor,
    CaffeInputTransformerFactory* transformer_factory);

  virtual ~CaffeGPUEvaluatorConstructor();

  virtual int get_number_of_devices() override;

  virtual DeviceType get_input_buffer_type() override;

  virtual DeviceType get_output_buffer_type() override;

  virtual int get_number_of_outputs() override;

  virtual std::vector<std::string> get_output_names() override;

  virtual std::vector<size_t> get_output_element_sizes() override;

  virtual char* new_input_buffer(const EvaluatorConfig& config) override;

  virtual void delete_input_buffer(
    const EvaluatorConfig& config,
    char* buffer) override;

  virtual std::vector<char*> new_output_buffers(
    const EvaluatorConfig& config,
    int num_inputs) override;

  virtual void delete_output_buffers(
    const EvaluatorConfig& config,
    std::vector<char*> buffers) override;

  virtual Evaluator* new_evaluator(const EvaluatorConfig& config) override;

private:
  NetDescriptor net_descriptor_;
  CaffeInputTransformerFactory* transformer_factory_;
};

}
