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
namespace internal {

struct SaveThreadArgs {
  // Uniform arguments
  std::string job_name;
  const std::vector<IOItem>& io_items;

  // Per worker arguments
  int id;
  storehouse::StorageConfig* storage_config;
  Profiler& profiler;

  // Queues for communicating work
  Queue<EvalWorkEntry>& input_work;
  std::atomic<i64>& retired_items;
};

void* save_thread(void* arg);

}
}
