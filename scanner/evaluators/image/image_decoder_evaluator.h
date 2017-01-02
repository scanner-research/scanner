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

#include <memory>

namespace scanner {

class ImageDecoderEvaluator : public Evaluator {
 public:
  ImageDecoderEvaluator(const EvaluatorConfig& config, DeviceType device_type);

  void configure(const BatchConfig& config) override;

  void evaluate(const BatchedColumns& input_columns,
                BatchedColumns& output_columns) override;

 private:
  DeviceType device_type_;
  i32 device_id_;
  i32 frame_width_;
  i32 frame_height_;
};

class ImageDecoderEvaluatorFactory : public EvaluatorFactory {
 public:
  ImageDecoderEvaluatorFactory(DeviceType device_type);

  EvaluatorCapabilities get_capabilities() override;

  std::vector<std::string> get_output_columns(
      const std::vector<std::string>& input_columns) override;

  Evaluator* new_evaluator(const EvaluatorConfig& config) override;

 private:
  DeviceType device_type_;
};
}
