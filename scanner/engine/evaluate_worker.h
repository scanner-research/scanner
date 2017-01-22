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

///////////////////////////////////////////////////////////////////////////////
/// Worker thread arguments
struct PreEvaluateThreadArgs {
  // Uniform arguments
  const std::vector<IOItem>& io_items;
  i32 warmup_count;

  // Per worker arguments
  int id;
  Profiler& profiler;

  // Queues for communicating work
  Queue<EvalWorkEntry>& input_work;
  Queue<EvalWorkEntry>& output_work;
};

struct EvaluateThreadArgs {
  // Uniform arguments
  const std::vector<IOItem>& io_items;
  i32 warmup_count;

  // Per worker arguments
  int id;
  std::vector<KernelFactory*> kernel_factories;
  std::vector<KernelConfig> kernel_configs;
  Profiler& profiler;

  // Queues for communicating work
  Queue<EvalWorkEntry>& input_work;
  Queue<EvalWorkEntry>& output_work;
};

struct PostEvaluateThreadArgs {
  // Uniform arguments
  const std::vector<IOItem>& io_items;
  i32 warmup_count;

  // Per worker arguments
  int id;
  Profiler& profiler;

  // Queues for communicating work
  Queue<EvalWorkEntry>& input_work;
  Queue<EvalWorkEntry>& output_work;
};

void* pre_evaluate_thread(void* arg);

void* evaluate_thread(void* arg);

void* post_evaluate_thread(void* arg);

}
}
