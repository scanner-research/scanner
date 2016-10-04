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

class FacenetParserEvaluator : public Evaluator {
public:
 FacenetParserEvaluator(const EvaluatorConfig& config, DeviceType device_type,
                        i32 device_id, double threshold,
                        bool forward_input = false);

 void configure(const DatasetItemMetadata& metadata) override;

 void evaluate(const std::vector<std::vector<u8*>>& input_buffers,
               const std::vector<std::vector<size_t>>& input_sizes,
               std::vector<std::vector<u8*>>& output_buffers,
               std::vector<std::vector<size_t>>& output_sizes) override;

protected:
  std::vector<BoundingBox> nms(const std::vector<BoundingBox>& boxes,
                               f32 overlap);

  EvaluatorConfig config_;
  DeviceType device_type_;
  i32 device_id_;
  bool forward_input_;

  i32 num_templates_;
  i32 net_input_width_;
  i32 net_input_height_;
  i32 cell_width_;
  i32 cell_height_;
  i32 grid_width_;
  i32 grid_height_;
  std::vector<std::vector<f32>> templates_;
  std::vector<i32> feature_vector_lengths_;
  std::vector<size_t> feature_vector_sizes_;

  double threshold_;

  DatasetItemMetadata metadata_;

};

class FacenetParserEvaluatorFactory : public EvaluatorFactory {
 public:
  FacenetParserEvaluatorFactory(DeviceType device_type, double threshold,
                                bool forward_input = false);

  EvaluatorCapabilities get_capabilities() override;

  std::vector<std::string> get_output_names() override;

  Evaluator* new_evaluator(const EvaluatorConfig& config) override;

 private:
  DeviceType device_type_;
  double threshold_;
  bool forward_input_;
};
}  // end namespace scanner
