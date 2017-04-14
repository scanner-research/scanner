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

#include <grpc/support/log.h>
#include "scanner/engine/rpc.grpc.pb.h"
#include "scanner/engine/runtime.h"
#include "scanner/engine/sampler.h"
#include "scanner/util/progress_bar.h"
#include "scanner/util/util.h"

#include <mutex>
#include <thread>

namespace scanner {
namespace internal {

class MasterImpl final : public proto::Master::Service {
 public:
  MasterImpl(DatabaseParameters& params);

  ~MasterImpl();

  // Expects context->peer() to return a string in the format
  // ipv4:<peer_address>:<random_port>
  // Returns the <peer_address> from the above format.
  static std::string get_worker_address_from_grpc_context(
    grpc::ServerContext* context);

  grpc::Status RegisterWorker(grpc::ServerContext* context,
                              const proto::WorkerParams* worker_info,
                              proto::Registration* registration);

  grpc::Status ActiveWorkers(grpc::ServerContext* context,
                             const proto::Empty* empty,
                             proto::RegisteredWorkers* registered_workers);

  grpc::Status IngestVideos(grpc::ServerContext* context,
                            const proto::IngestParameters* params,
                            proto::IngestResult* result);

  grpc::Status NextWork(grpc::ServerContext* context,
                        const proto::NodeInfo* node_info,
                        proto::NewWork* new_work);

  grpc::Status NewJob(grpc::ServerContext* context,
                      const proto::JobParameters* job_params,
                      proto::Result* job_result);

  grpc::Status Ping(grpc::ServerContext* context, const proto::Empty* empty1,
                    proto::Empty* empty2);

  grpc::Status GetOpInfo(grpc::ServerContext* context,
                         const proto::OpInfoArgs* op_info_args,
                         proto::OpInfo* op_info);

  grpc::Status LoadOp(grpc::ServerContext* context,
                      const proto::OpPath* op_path, Result* result);

  grpc::Status Shutdown(grpc::ServerContext* context, const proto::Empty* empty,
                        Result* result);

  grpc::Status PokeWatchdog(grpc::ServerContext* context,
                            const proto::Empty* empty, proto::Empty* result);

  void start_watchdog(grpc::Server* server, i32 timeout_ms=50000);

 private:
  std::thread watchdog_thread_;
  std::atomic<bool> watchdog_awake_;
  std::vector<std::unique_ptr<proto::Worker::Stub>> workers_;
  std::vector<std::string> addresses_;
  Flag trigger_shutdown_;
  DatabaseParameters db_params_;
  storehouse::StorageBackend* storage_;
  std::map<std::string, TableMetadata> table_metas_;
  proto::JobParameters job_params_;
  ProgressBar* bar_;

  i64 total_samples_used_;
  i64 total_samples_;

  std::mutex work_mutex_;
  i64 next_task_;
  i64 num_tasks_;
  std::unique_ptr<TaskSampler> task_sampler_;
  i64 samples_left_;
  Result task_result_;
};
}
}
