/* Copyright 2016 Carnegie Mellon University, NVIDIA Corporation
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

#include "scanner/eval/evaluator_factory.h"
#include "scanner/util/common.h"

#include <functional>
#include <vector>

namespace scanner {

struct PointSamples {
  i32 video_index;
  std::vector<i32> frames;
};

struct SequenceSamples {
  i32 video_index;
  std::vector<Interval> intervals;
};

/**
 * @brief Defines evaluators and a sampling pattern to run over a dataset.
 *
 * A pipeline is a sequence, or chain, of evaluators which execute over a stream
 * of video data. A sampling pattern can be specified that selects a subset of
 * frames from the videos in a given dataset. The chain of evaluators is
 * specified by the "evaluator_factories" variable. The sampling pattern
 * defaults to reading all frames of all videos in the dataset (Sampling::All),
 * but can be refined to grab only every n-th frame (Sampling::Stride), select a
 * subset of individual frames (Sampling::Gather), or select in dense sequences
 * of consecutive frames (Sampling::SequenceGather).
 */
struct PipelineDescription {
  // Columns to grab from the input job
  std::vector<std::string> input_columns;

  Sampling sampling = Sampling::All;
  // For strided sampling
  i32 stride;
  // For gather sampling
  std::vector<PointSamples> gather_points;
  // For sequence gather sampling
  std::vector<SequenceSamples> gather_sequences;

  // The chain of evaluators which will be executed over the input
  std::vector<std::unique_ptr<EvaluatorFactory>> evaluator_factories;
};

struct DatasetItemMetadata {
 public:
  DatasetItemMetadata(i32 frames, i32 width, i32 height);

  i32 frames() const;
  i32 width() const;
  i32 height() const;

 private:
  i32 frames_;
  i32 width_;
  i32 height_;
};

using PipelineGeneratorFn = std::function<PipelineDescription(
    const DatasetMetadata&, const std::vector<DatasetItemMetadata>&)>;

bool add_pipeline(std::string name, PipelineGeneratorFn fn);

PipelineGeneratorFn get_pipeline(const std::string& name);

#define REGISTER_PIPELINE(name, fn) \
  static bool dummy_##name = add_pipeline(#name, fn);
}
