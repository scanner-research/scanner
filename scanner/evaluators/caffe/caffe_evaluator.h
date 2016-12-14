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
    std::function<void(const InputFormat &descriptor, caffe::Net<float> *net)>;

class CaffeEvaluator : public Evaluator {
 public:
   CaffeEvaluator(const EvaluatorConfig &config, DeviceType device_type,
                  i32 device_id, const NetDescriptor &descriptor,
                  i32 batch_size, bool forward_input = false,
                  CustomNetConfiguration net_config = nullptr);

   void configure(const InputFormat &descriptor) override;

   void evaluate(const BatchedColumns &input_columns,
                 BatchedColumns &output_columns) override;

 protected:
  void set_device();

  EvaluatorConfig config_;
  DeviceType device_type_;
  i32 device_id_;
  NetDescriptor descriptor_;
  i32 batch_size_;
  bool forward_input_;
  CustomNetConfiguration net_config_;
  std::unique_ptr<caffe::Net<float>> net_;

  InputFormat metadata_;
};

class CaffeEvaluatorFactory : public EvaluatorFactory {
 public:
  CaffeEvaluatorFactory(DeviceType device_type,
                        const NetDescriptor& net_descriptor, i32 batch_size,
                        bool forward_input = false,
                        CustomNetConfiguration net_config = nullptr);

  EvaluatorCapabilities get_capabilities() override;

  std::vector<std::string> get_output_names() override;

  Evaluator* new_evaluator(const EvaluatorConfig& config) override;

 private:
  DeviceType device_type_;
  NetDescriptor net_descriptor_;
  i32 batch_size_;
  bool forward_input_;
  CustomNetConfiguration net_config_;
};
}  // end namespace scanner
