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
#include "scanner/evaluators/caffe/caffe_input_transformer_factory.h"
#include "scanner/evaluators/caffe/net_descriptor.h"

#include <memory>
#include <vector>

namespace scanner {

class CaffeEvaluator : public Evaluator {
 public:
  CaffeEvaluator(const EvaluatorConfig& config, DeviceType device_type,
                 const NetDescriptor& descriptor,
                 CaffeInputTransformer* transformer, i32 device_id);

  void configure(const DatasetItemMetadata& metadata) override;

  void evaluate(i32 input_count, u8* input_buffer,
                std::vector<std::vector<u8*>>& output_buffers,
                std::vector<std::vector<size_t>>& output_sizes) override;

 protected:
  EvaluatorConfig config_;
  DeviceType device_type_;
  NetDescriptor descriptor_;
  std::unique_ptr<CaffeInputTransformer> transformer_;
  i32 device_id_;
  std::unique_ptr<caffe::Net<float>> net_;

  DatasetItemMetadata metadata_;
};

class CaffeEvaluatorFactory : public EvaluatorFactory {
 public:
  CaffeEvaluatorFactory(DeviceType device_type,
                        const NetDescriptor& net_descriptor,
                        CaffeInputTransformerFactory* transformer_factory);

  EvaluatorCapabilities get_capabilities() override;

  i32 get_number_of_outputs() override;

  std::vector<std::string> get_output_names() override;

  Evaluator* new_evaluator(const EvaluatorConfig& config) override;

 private:
  DeviceType device_type_;
  NetDescriptor net_descriptor_;
  std::unique_ptr<CaffeInputTransformerFactory> transformer_factory_;
};
}  // end namespace scanner
