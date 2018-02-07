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
#include "scanner/evaluators/types.pb.h"

#include "struck/Tracker.h"

#include <memory>
#include <random>
#include <vector>

namespace scanner {

class TrackerEvaluator : public Evaluator {
 public:
  TrackerEvaluator(const EvaluatorConfig& config, DeviceType device_type,
                   i32 device_id, i32 warmup_count, i32 max_tracks);

  void configure(const BatchConfig& config) override;

  void reset() override;

  void evaluate(const BatchedElements& input_columns,
                BatchedElements& output_columns) override;

 protected:
  float iou(const BoundingBox& bl, const BoundingBox& br);

  const f32 IOU_THRESHOLD = 0.4;
  const f64 TRACK_SCORE_THRESHOLD = 0.1;

  EvaluatorConfig config_;
  DeviceType device_type_;
  i32 device_id_;
  i32 warmup_count_;
  i32 max_tracks_;

  InputFormat metadata_;

  i32 next_tracker_id_;
  std::mt19937 gen{std::random_device{}()};
  std::uniform_int_distribution<i32> unif;
  struct Track {
    i32 id;
    BoundingBox box;
    std::unique_ptr<struck::Config> config;
    std::unique_ptr<struck::Tracker> tracker;
    i32 frames_since_last_detection;
  };
  std::vector<Track> tracks_;
};

class TrackerEvaluatorFactory : public EvaluatorFactory {
 public:
  TrackerEvaluatorFactory(DeviceType device_type, i32 warmup_count,
                          i32 max_tracks);

  EvaluatorCapabilities get_capabilities() override;

  std::vector<std::string> get_output_columns(
      const std::vector<std::string>& input_columns) override;

  Evaluator* new_evaluator(const EvaluatorConfig& config) override;

 private:
  DeviceType device_type_;
  i32 warmup_count_;
  i32 max_tracks_;
};
}  // end namespace scanner
