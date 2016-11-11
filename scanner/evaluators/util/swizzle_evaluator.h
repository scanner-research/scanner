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
#include <vector>

namespace scanner {

class SwizzleEvaluator : public Evaluator {
 public:
  SwizzleEvaluator(const EvaluatorConfig& config, DeviceType device_type,
                   i32 device_id, const std::vector<i32>& output_to_input_idx);

  void configure(const InputFormat& metadata) override;

  void evaluate(const std::vector<std::vector<u8*>>& input_buffers,
                const std::vector<std::vector<size_t>>& input_sizes,
                std::vector<std::vector<u8*>>& output_buffers,
                std::vector<std::vector<size_t>>& output_sizes) override;

 protected:
  EvaluatorConfig config_;
  DeviceType device_type_;
  i32 device_id_;
  std::vector<i32> output_to_input_idx_;

  InputFormat metadata_;
};

class SwizzleEvaluatorFactory : public EvaluatorFactory {
 public:
  SwizzleEvaluatorFactory(DeviceType device_type,
                          const std::vector<i32>& output_to_input_idx,
                          const std::vector<std::string>& output_names);

  EvaluatorCapabilities get_capabilities() override;

  std::vector<std::string> get_output_names() override;

  Evaluator* new_evaluator(const EvaluatorConfig& config) override;

 private:
  DeviceType device_type_;
  std::vector<i32> output_to_input_idx_;
  std::vector<std::string> output_names_;
};
}  // end namespace scanner
