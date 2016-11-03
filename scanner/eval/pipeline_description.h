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

struct Interval {
  i32 start;
  i32 end;
};

struct SequenceSamples {
  i32 video_index;
  std::vector<Interval> intervals;
};

enum class Sampling {
  All,
  Strided,
  Gather,
  SequenceGather,
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
  std::vector<std::unique_ptr<EvaluatorFactory>> evaluator_factories;

  Sampling sampling = Sampling::All;
  // For strided sampling
  i32 stride;
  // For gather sampling
  std::vector<PointSamples> gather_points;
  // For sequence gather sampling
  std::vector<SequenceSamples> gather_sequences;
};

bool add_pipeline(std::string name,
                  std::function<PipelineDescription(void)> fn);

std::function<PipelineDescription(void)> get_pipeline(const std::string& name);

#define REGISTER_PIPELINE(name, fn) \
  static bool dummy_##name = add_pipeline(#name, fn);
}
