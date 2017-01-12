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

#include "scanner/engine/runtime.h"
#include "scanner/util/common.h"
#include "scanner/util/queue.h"

namespace scanner {

struct LoadThreadArgs {
  // Uniform arguments
  const DatasetMetadata& dataset;
  const std::map<i32, JobMetadata>& job_meta;
  const std::map<i32, VideoMetadata>& video_meta;
  const std::vector<ImageFormatGroupMetadata>& image_meta;
  const std::vector<IOItem>& io_items;
  i32 warmup_count;

  // Per worker arguments
  int id;
  storehouse::StorageConfig* storage_config;
  Profiler& profiler;

  // Queues for communicating work
  Queue<LoadWorkEntry>& load_work;  // in
  Queue<EvalWorkEntry>& eval_work;  // out
};

void* load_thread(void* arg);

}
