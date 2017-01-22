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

#include "scanner/util/memory.h"
#include "storehouse/storage_backend.h"

#include <grpc++/server.h>
#include "scanner/engine/rpc.grpc.pb.h"

#include <string>

namespace scanner {

struct DatabaseParameters {
  storehouse::StorageConfig* storage_config;
  MemoryPoolConfig memory_pool_config;
  std::string db_path;
};

struct ServerState {
  std::unique_ptr<grpc::Server> server;
  std::unique_ptr<grpc::Service> service;
};

ServerState start_master(DatabaseParameters &params, bool block = true);

ServerState start_worker(DatabaseParameters &params,
                           const std::string &master_address,
                           bool block = true);

struct JobParameters {
  proto::TaskSet task_set;
};

void new_job(JobParameters& params);

}
