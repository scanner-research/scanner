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

#include "scanner/api/op.h"
#include "scanner/util/memory.h"
#include "storehouse/storage_backend.h"

#include <grpc++/server.h>
#include "scanner/engine/rpc.grpc.pb.h"

#include <string>

namespace scanner {

//! Description of resources for a given machine.
struct MachineParameters {
  i32 num_cpus;
  i32 num_load_workers;
  i32 num_save_workers;
  std::vector<i32>
      gpu_ids;  //!< List of CUDA device IDs that Scanner should use.
};

//! Pick smart defaults for the current machine.
MachineParameters default_machine_params();

//! Info about a video that fails to ingest.
struct FailedVideo {
  std::string path;
  std::string message;
};

//! Main entry point into Scanner.
class Database {
 public:
  Database(storehouse::StorageConfig* storage_config,
           const std::string& db_path, const std::string& master_address);

  Result start_master(const MachineParameters& params, const std::string& port,
                      bool watchdog = true,
                      bool prefetch_table_metadata = true,
                      i64 no_workers_timeout = 30);

  Result start_worker(const MachineParameters& params, const std::string& port,
                      bool watchdog = true);

  Result ingest_videos(const std::vector<std::string>& table_names,
                       const std::vector<std::string>& paths,
                       bool inplace,
                       std::vector<FailedVideo>& failed_videos);

  Result new_table(const std::string& table_name,
                   const std::vector<std::string>& columns,
                   const std::vector<std::vector<std::string>>& rows);

  Result delete_table(const std::string& table_name);

  Result shutdown_master();

  Result shutdown_worker();

  Result wait_for_server_shutdown();

  Result destroy_database();

 protected:
  bool database_exists();

  struct ServerState {
    std::unique_ptr<grpc::Server> server;
    std::shared_ptr<grpc::Service> service;
  };

 private:
  storehouse::StorageConfig* storage_config_;
  std::unique_ptr<storehouse::StorageBackend> storage_;
  std::string db_path_;
  std::string master_address_;

  std::unique_ptr<ServerState> master_state_;
  std::vector<std::unique_ptr<ServerState>> worker_states_;
};
}
