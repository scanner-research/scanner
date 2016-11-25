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
#include "scanner/eval/pipeline_description.h"
#include "scanner/video/video_decoder.h"

#include "storehouse/storage_backend.h"

#include <string>

namespace scanner {

///////////////////////////////////////////////////////////////////////////////
/// Work structs - structs used to exchange data between workers during
///   execution of the run command.
struct WorkItem {
  i32 video_index;
  i64 item_index;
  i64 item_id;
  i64 next_item_id;
  i32 rows_from_start;
};

struct LoadWorkEntry {
  i32 work_item_index;
  // union {
  // For no sampling
  Interval interval;
  // For stride
  StridedInterval strided_interval;
  // For gather
  std::vector<i32> gather_points;
  // For sequence gather
  std::vector<StridedInterval> gather_sequences;
  //};
};

struct EvalWorkEntry {
  i32 work_item_index;
  std::vector<std::string> column_names;
  BatchedColumns columns;
  DeviceType buffer_type;
  i32 buffer_device_id;
  bool video_decode_item;
};

///////////////////////////////////////////////////////////////////////////////
void run_job(storehouse::StorageConfig* storage_config,
             const std::string& dataset_name, const std::string& in_job_name,
             PipelineGeneratorFn pipeline_gen_fn,
             const std::string& out_job_name);
}
