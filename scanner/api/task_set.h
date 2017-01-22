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

#include "scanner/api/evaluator.h"
#include "scanner/util/common.h"
#include "scanner/util/profiler.h"

#include "scanner/metadata.pb.h"

#include <vector>

namespace scanner {

struct TaskSet {
  std::string job_name;
  std::vector<scanner::proto::Task> tasks;
  Evaluator* output_evaluator;
};

proto::TaskSet consume_task_set(TaskSet& task_set);

}
