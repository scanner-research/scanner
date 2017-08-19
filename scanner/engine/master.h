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

  grpc::Status IsJobDone(grpc::ServerContext* context,
                         const proto::Empty* empty,
                         proto::JobResult* job_result);

  grpc::Status Ping(grpc::ServerContext* context, const proto::Empty* empty1,
                    proto::Empty* empty2);

  grpc::Status GetOpInfo(grpc::ServerContext* context,
                         const proto::OpInfoArgs* op_info_args,
                         proto::OpInfo* op_info);

  grpc::Status LoadOp(grpc::ServerContext* context,
                      const proto::OpPath* op_path, Result* result);

  grpc::Status RegisterOp(grpc::ServerContext* context,
                          const proto::OpRegistration* op_registration,
                          proto::Result* result);

  grpc::Status RegisterPythonKernel(
      grpc::ServerContext* context,
      const proto::PythonKernelRegistration* python_kernel,
      proto::Result* result);

  grpc::Status Shutdown(grpc::ServerContext* context, const proto::Empty* empty,
                        Result* result);

  grpc::Status PokeWatchdog(grpc::ServerContext* context,
                            const proto::Empty* empty, proto::Empty* result);

  void start_watchdog(grpc::Server* server, bool enable_timeout,
                      i32 timeout_ms = 50000);

 private:
  void start_job_processor();

  void stop_job_processor();

  bool process_job(const proto::JobParameters* job_params,
                   proto::Result* job_result);

  void start_worker_pinger();

  void stop_worker_pinger();

  void start_job_on_worker(i32 node_id, const std::string& address);

  void stop_job_on_worker(i32 node_id);

  void remove_worker(i32 node_id);


  std::thread watchdog_thread_;
  std::atomic<bool> watchdog_awake_;
  i32 next_worker_id_ = 0;
  std::map<i32, bool> worker_active_;
  std::map<i32, std::unique_ptr<proto::Worker::Stub>> workers_;
  std::map<i32, std::string> worker_addresses_;
  Flag trigger_shutdown_;
  DatabaseParameters db_params_;
  storehouse::StorageBackend* storage_;
  DatabaseMetadata meta_;
  std::unique_ptr<TableMetaCache> table_metas_;
  proto::JobParameters job_params_;
  std::unique_ptr<ProgressBar> bar_;
  std::vector<std::string> so_paths_;
  std::vector<proto::OpRegistration> op_registrations_;
  std::vector<proto::PythonKernelRegistration> py_kernel_registrations_;

  i64 total_samples_used_;
  i64 total_samples_;

  // True if the master is executing a job
  std::mutex active_mutex_;
  std::condition_variable active_cv_;
  bool active_job_ = false;

  // True if all work for job is done
  std::mutex finished_mutex_;
  std::condition_variable finished_cv_;
  bool finished_ = true;
  Result job_result_;

  std::thread job_processor_thread_;
  // Manages modification of all of the below structures
  std::mutex work_mutex_;
  // Outstanding set of generated task samples that should be processed
  std::deque<std::tuple<i64, i64>> unallocated_task_samples_;
  // The next task to use to generate task samples
  i64 next_task_;
  // Total number of tasks
  i64 num_tasks_;
  // Cache of task samplers for active tasks (in unallocated or current task)
  std::map<i64, std::unique_ptr<TaskSampler>> task_samplers_;
  // # of samples that are left before the task sampler is no longer active
  std::map<i64, i64> task_sampler_samples_left_;
  // Next sample index in the current task
  i64 next_sample_;
  // Total samples in the current task
  i64 num_samples_;
  Result task_result_;
  // Worker id -> (task_id, sample_id)
  std::map<i64, std::set<std::tuple<i64, i64>>> active_task_samples_;
  // Track assignment of tasks to worker for this job
  struct WorkerHistory {
    timepoint_t start_time;
    timepoint_t end_time;
    i64 tasks_assigned;
    i64 tasks_retired;
  };
  std::map<i64, WorkerHistory> worker_histories_;
  std::map<i32, bool> unfinished_workers_;

  // Worker connections
  std::map<std::string, i32> local_ids_;
  std::map<std::string, i32> local_totals_;
  grpc::CompletionQueue cq_;
  std::map<i32, std::unique_ptr<grpc::ClientContext>> client_contexts_;
  std::map<i32, std::unique_ptr<grpc::Status>> statuses_;
  std::map<i32, std::unique_ptr<proto::Result>> replies_;
  std::map<i32, std::unique_ptr<grpc::ClientAsyncResponseReader<proto::Result>>>
      rpcs_;
};
}
}
