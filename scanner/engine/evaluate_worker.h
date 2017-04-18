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

#include "scanner/engine/kernel_factory.h"
#include "scanner/engine/runtime.h"
#include "scanner/util/common.h"
#include "scanner/util/queue.h"

namespace scanner {
namespace internal {

void move_if_different_address_space(Profiler& profiler,
                                     DeviceHandle current_handle,
                                     DeviceHandle target_handle,
                                     BatchedColumns& columns);

///////////////////////////////////////////////////////////////////////////////
/// Worker thread arguments
struct PreEvaluateThreadArgs {
  // Uniform arguments
  i32 node_id;
  i32 num_cpus;
  const proto::JobParameters* job_params;

  // Per worker arguments
  i32 id;
  DeviceHandle device_handle;
  Profiler& profiler;

  // Queues for communicating work
  Queue<std::tuple<IOItem, EvalWorkEntry>>& input_work;
  Queue<std::tuple<IOItem, EvalWorkEntry>>& output_work;
};

struct EvaluateThreadArgs {
  // Uniform arguments
  i32 node_id;
  const proto::JobParameters* job_params;

  // Per worker arguments
  i32 ki;
  i32 kg;
  std::vector<std::tuple<KernelFactory*, Kernel::Config>> kernel_factories;
  std::vector<std::vector<std::tuple<i32, std::string>>> live_columns;
  // Discarded after kernel use
  std::vector<std::vector<i32>> dead_columns;
  // Discarded immediately after kernel execute
  std::vector<std::vector<i32>> unused_outputs;
  // Index in columns for inputs
  std::vector<std::vector<i32>> column_mapping;
  Profiler& profiler;
  proto::Result& result;

  // Queues for communicating work
  Queue<std::tuple<IOItem, EvalWorkEntry>>& input_work;
  Queue<std::tuple<IOItem, EvalWorkEntry>>& output_work;
};

struct PostEvaluateThreadArgs {
  // Uniform arguments
  i32 node_id;

  // Per worker arguments
  i32 id;
  Profiler& profiler;
  // Index in columns for inputs
  std::vector<i32> column_mapping;
  std::vector<Column> columns;

  // Queues for communicating work
  Queue<std::tuple<IOItem, EvalWorkEntry>>& input_work;
  Queue<std::tuple<IOItem, EvalWorkEntry>>& output_work;
};

void* pre_evaluate_thread(void* arg);

void* evaluate_thread(void* arg);

void* post_evaluate_thread(void* arg);
}
}
