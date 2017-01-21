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

#include "scanner/api/run.h"
#include "scanner/engine/runtime.h"
#include "scanner/engine/evaluator_registry.h"
#include "scanner/engine/kernel_registry.h"
#include "scanner/engine/save_worker.h"
#include "scanner/engine/evaluate_worker.h"
#include "scanner/engine/load_worker.h"
#include "scanner/engine/db.h"
#include "scanner/engine/rpc.grpc.pb.h"

namespace scanner {
namespace internal {

}

using namespace internal;

void run_job(JobParameters& params) {
  EvaluatorRegistry* evaluator_registry = get_evaluator_registry();
  KernelRegistry* kernel_registry = get_kernel_registry();
  for (auto& evaluator : params.task_set.evaluators()) {
    const std::string& name = evaluator.name();
    EvaluatorInfo* evaluator_info =
      evaluator_registry->get_evaluator_info(name);
    KernelFactory* kernel_factory =
      kernel_registry->get_kernel(name, evaluator.device_type());
  }
}

}
