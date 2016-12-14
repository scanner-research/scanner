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

class ModelDescriptor;

class CPM2ParserEvaluator : public Evaluator {
public:
  CPM2ParserEvaluator(const EvaluatorConfig &config, DeviceType device_type,
                      i32 device_id, bool forward_input = false);

  void configure(const InputFormat &metadata) override;

  void evaluate(const BatchedColumns &input_columns,
                BatchedColumns &output_columns) override;

 protected:
   int connect_limbs(std::vector<std::vector<double>> &subset,
                     std::vector<std::vector<std::vector<double>>> &connection,
                     const float *heatmap_pointer, const float *peaks,
                     float *joints);

   EvaluatorConfig config_;
   DeviceType device_type_;
   i32 device_id_;
   f32 threshold_ = 0.5f;
   bool forward_input_;

   std::unique_ptr<ModelDescriptor> modeldesc;
   InputFormat metadata_;
   // The maximum number of joint peaks from the nms output layer
   i32 cell_size_ = 8;
   i32 box_size_ = 368;
   i32 resize_width_;
   i32 resize_height_;
   i32 width_padding_;
   i32 padded_width_;
   i32 net_input_width_;
   i32 net_input_height_;
   i32 feature_width_;
   i32 feature_height_;
   i32 feature_channels_;

   const int max_people_ = 96;
   const int max_num_parts_ = 70;
   const int max_peaks_ = 20;
   const int num_joints_ = 15;
   int connect_min_subset_cnt_ = 3;
   float connect_min_subset_score_ = 0.4;
   float connect_inter_threshold_ = 0.01;
   int connect_inter_min_above_threshold_ = 8;
   std::vector<float> joints_;
};

class CPM2ParserEvaluatorFactory : public EvaluatorFactory {
public:
  CPM2ParserEvaluatorFactory(DeviceType device_type,
                             bool forward_input = false);

  EvaluatorCapabilities get_capabilities() override;

  std::vector<std::string> get_output_names() override;

  Evaluator *new_evaluator(const EvaluatorConfig &config) override;

 private:
  DeviceType device_type_;
  bool forward_input_;
};
}  // end namespace scanner
