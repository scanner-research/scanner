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

  grpc::Status UnregisterWorker(grpc::ServerContext* context,
                                const proto::NodeInfo* node_info,
                                proto::Empty* empty);

  grpc::Status ActiveWorkers(grpc::ServerContext* context,
                             const proto::Empty* empty,
                             proto::RegisteredWorkers* registered_workers);

  grpc::Status IngestVideos(grpc::ServerContext* context,
                            const proto::IngestParameters* params,
                            proto::IngestResult* result);

  grpc::Status NextWork(grpc::ServerContext* context,
                        const proto::NodeInfo* node_info,
                        proto::NewWork* new_work);

  grpc::Status FinishedWork(grpc::ServerContext* context,
                            const proto::FinishedWorkParameters* params,
                            proto::Empty* empty);

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

  void start_watchdog(grpc::Server* server, i32 timeout_ms = 50000);

 private:
  void remove_worker(i32 node_id);

  std::thread watchdog_thread_;
  std::atomic<bool> watchdog_awake_;
  i32 next_worker_id_ = 0;
  std::map<i32, std::unique_ptr<proto::Worker::Stub>> workers_;
  std::map<i32, std::string> worker_addresses_;
  Flag trigger_shutdown_;
  DatabaseParameters db_params_;
  storehouse::StorageBackend* storage_;
  DatabaseMetadata meta_;
  std::unique_ptr<TableMetaCache> table_metas_;
  proto::JobParameters job_params_;
  std::unique_ptr<ProgressBar> bar_;

  i64 total_samples_used_;
  i64 total_samples_;

  std::mutex work_mutex_;
  std::deque<std::tuple<i64, i64>> unallocated_task_samples_;
  i64 next_task_;
  i64 num_tasks_;
  std::map<i64, std::unique_ptr<TaskSampler>> task_samplers_;
  // Tracks how many samples are left before the task sampler can be
  // deallocated
  std::map<i64, i64> task_sampler_samples_left_;
  i64 next_sample_;
  i64 num_samples_;
  Result task_result_;
  // Worker id -> (task_id, sample_id)
  std::map<i64, std::set<std::tuple<i64, i64>>> active_task_samples_;
};
}
}
