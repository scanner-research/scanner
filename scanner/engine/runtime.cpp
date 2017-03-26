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

#include "scanner/engine/runtime.h"
#include "scanner/engine/master.h"
#include "scanner/engine/worker.h"

namespace scanner {
namespace internal {

MasterImpl* get_master_service(DatabaseParameters& param) {
  return new MasterImpl(param);
}

WorkerImpl* get_worker_service(DatabaseParameters& params,
                               const std::string& master_address,
                               const std::string& worker_port) {
  return new WorkerImpl(params, master_address, worker_port);
}
}
}
