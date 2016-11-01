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

#include "scanner/eval/evaluator.h"
#include "scanner/eval/evaluator_factory.h"
#include "scanner/video/video_decoder.h"

#include "storehouse/storage_backend.h"

#include <string>

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

///////////////////////////////////////////////////////////////////////////////
void run_job(storehouse::StorageConfig* storage_config,
             PipelineDescription& pipeline_description,
             const std::string& job_name, const std::string &dataset_name);
}
