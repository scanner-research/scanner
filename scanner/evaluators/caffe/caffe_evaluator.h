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

#include <memory>
#include <vector>

namespace scanner {

using CustomNetConfiguration =
    std::function<void(const BatchConfig& config, caffe::Net<float>* net)>;

class CaffeEvaluator : public Evaluator {
 public:
  CaffeEvaluator(const EvaluatorConfig& config, DeviceType device_type,
                 i32 device_id, const NetDescriptor& descriptor, i32 batch_size,
                 CustomNetConfiguration net_config = nullptr);

  void configure(const BatchConfig& config) override;

  void evaluate(const BatchedColumns& input_columns,
                BatchedColumns& output_columns) override;

 protected:
  void set_device();

  EvaluatorConfig eval_config_;
  DeviceType device_type_;
  i32 device_id_;
  NetDescriptor descriptor_;
  i32 batch_size_;
  CustomNetConfiguration net_config_;
  std::unique_ptr<caffe::Net<float>> net_;

  i32 frame_width;
  i32 frame_height;
};

class CaffeEvaluatorFactory : public EvaluatorFactory {
 public:
  CaffeEvaluatorFactory(DeviceType device_type,
                        const NetDescriptor& net_descriptor, i32 batch_size,
                        CustomNetConfiguration net_config = nullptr);

  EvaluatorCapabilities get_capabilities() override;

  std::vector<std::string> get_output_columns(
      const std::vector<std::string>& input_columns) override;

  Evaluator* new_evaluator(const EvaluatorConfig& config) override;

 private:
  DeviceType device_type_;
  NetDescriptor net_descriptor_;
  i32 batch_size_;
  CustomNetConfiguration net_config_;
};
}  // end namespace scanner
