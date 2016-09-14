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
#include <opencv2/stitching.hpp>
#include <opencv2/imgproc.hpp>

namespace scanner {

class CaffeCPUEvaluator : public Evaluator {
public:
  CaffeCPUEvaluator(
    EvaluatorConfig config,
    NetDescriptor descriptor,
    int device_id);

  virtual ~CaffeGPUEvaluator();

  virtual void evaluate(
    const DatasetItemMetadata& metadata,
    char* input_buffer,
    std::vector<char*> output_buffers,
    int batch_size) override;

private:
  NetDescriptor descriptor_;
  int device_id_;
  std::unique_ptr<caffe::Net<float>> net_;
  cv::Mat mean_mat_; // mean image for input normalization

  std::vector<size_t> output_sizes_;

  cv::Mat input_mat;
  cv::Mat rgba_mat;
  cv::Mat rgb_mat;
  cv::Mat conv_input;
  cv::Mat conv_planar_input;
  cv::Mat float_conv_input;
  cv::Mat normed_input;
  cv::Mat scaled_input;
};

class CaffeCPUEvaluatorConstructor : public EvaluatorConstructor {
public:
  CaffeCPUEvaluatorConstructor(NetDescriptor net_descriptor);

  virtual ~CaffeCPUEvaluatorConstructor();

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
};

}
