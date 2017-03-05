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

struct MachineParameters {
  i32 num_cpus;
  i32 num_load_workers;
  i32 num_save_workers;
  std::vector<i32> gpu_ids;
};

MachineParameters default_machine_params();

struct TableSample {
  std::string table_name;
  std::vector<std::string> column_names;
  std::string sampling_function;
  std::vector<u8> sampling_args;
};

struct Task {
  std::string output_table_name;
  std::vector<TableSample> samples;
};

struct TaskSet {
  std::vector<Task> tasks;
  Op *output_op;
};

struct JobParameters {
  std::string job_name;
  TaskSet task_set;

  MemoryPoolConfig memory_pool_config;
  i32 pipeline_instances_per_node;
  i64 work_item_size;
};

struct FailedVideo {
  std::string path;
  std::string message;
};

class Database {
public:
  Database(storehouse::StorageConfig *storage_config,
           const std::string &db_path,
           const std::string &master_address,
           const std::string &master_port,
           const std::string &worker_port);

  Result start_master(const MachineParameters &params);

  Result start_worker(const MachineParameters &params);

  Result ingest_videos(const std::vector<std::string> &table_names,
                              const std::vector<std::string> &paths,
                              std::vector<FailedVideo> &failed_videos);

  // void ingest_images(storehouse::StorageConfig *storage_config,
  //                    const std::string &db_path, const std::string &table_name,
  //                    const std::vector<std::string> &paths);

  Result new_job(JobParameters &params);

  Result shutdown_master();

  Result shutdown_worker();

  Result wait_for_server_shutdown();

  Result destroy_database();

protected:
  bool database_exists();

  struct ServerState {
    std::unique_ptr<grpc::Server> server;
    std::unique_ptr<grpc::Service> service;
    std::unique_ptr<std::atomic<bool>> request_shutdown;
  };

private:
  storehouse::StorageConfig *storage_config_;
  std::unique_ptr<storehouse::StorageBackend> storage_;
  std::string db_path_;
  std::string master_address_;
  std::string master_port_;
  std::string worker_port_;

  ServerState master_state_;
  std::vector<ServerState> worker_states_;
};
}
