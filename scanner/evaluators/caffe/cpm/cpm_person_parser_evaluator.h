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
#include "scanner/util/opencv.h"

#include <memory>
#include <vector>

namespace scanner {

class CPMPersonParserEvaluator : public Evaluator {
 public:
  CPMPersonParserEvaluator(const EvaluatorConfig& config,
                           DeviceType device_type, i32 device_id,
                           bool forward_input = false);

  void configure(const InputFormat& metadata) override;

  void evaluate(const BatchedColumns& input_columns,
                BatchedColumns& output_columns) override;

 protected:
  EvaluatorConfig config_;
  DeviceType device_type_;
  i32 device_id_;
  f32 threshold_ = 0.5f;
  bool forward_input_;

  InputFormat metadata_;
  i32 cell_size_ = 8;
  i32 box_size_ = 368;
  i32 resize_width_;
  i32 resize_height_;
  i32 width_padding_;
  i32 net_input_width_;
  i32 net_input_height_;
  i32 feature_width_;
  i32 feature_height_;

  cv::Mat dilate_kernel_;
  cv::Mat resized_c_;
  cv::Mat max_c_;
};

class CPMPersonParserEvaluatorFactory : public EvaluatorFactory {
 public:
  CPMPersonParserEvaluatorFactory(DeviceType device_type,
                                  bool forward_input = false);

  EvaluatorCapabilities get_capabilities() override;

  std::vector<std::string> get_output_names() override;

  Evaluator* new_evaluator(const EvaluatorConfig& config) override;

 private:
  DeviceType device_type_;
  bool forward_input_;
};
}  // end namespace scanner
