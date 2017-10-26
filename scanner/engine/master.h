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
                      const proto::BulkJobParameters* job_params,
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

  bool process_job(const proto::BulkJobParameters* job_params,
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
  std::unique_ptr<ProgressBar> bar_;
  std::vector<std::string> so_paths_;
  std::vector<proto::OpRegistration> op_registrations_;
  std::vector<proto::PythonKernelRegistration> py_kernel_registrations_;
  proto::BulkJobParameters job_params_;

  i64 total_tasks_used_;
  i64 total_tasks_;

  // True if the master is executing a job
  std::mutex active_mutex_;
  std::condition_variable active_cv_;
  bool active_bulk_job_ = false;

  // True if all work for job is done
  std::mutex finished_mutex_;
  std::condition_variable finished_cv_;
  bool finished_ = true;
  Result job_result_;

  std::thread job_processor_thread_;
  // Manages modification of all of the below structures
  std::mutex work_mutex_;
  // Mapping from jobs to table ids
  std::map<i64, i64> job_to_table_id_;
  // Slice input rows for each job at each slice op
  std::vector<std::map<i64, i64>> slice_input_rows_per_job_;
  // Output rows for each job
  std::vector<i64> total_output_rows_per_job_;
  // All job task output rows
  std::vector<std::vector<std::vector<i64>>> job_tasks_;
  // Outstanding set of generated task samples that should be processed
  std::deque<std::tuple<i64, i64>> unallocated_job_tasks_;
  // The next job to use to generate tasks
  i64 next_job_;
  // Total number of jobs
  i64 num_jobs_;
  // Next sample index in the current task
  i64 next_task_;
  // Total samples in the current task
  i64 num_tasks_;
  Result task_result_;
  // Tracks tasks assigned to worker so they can be reassigned if the worker
  // fails
  // Worker id -> (job_id, task_id)
  std::map<i64, std::set<std::tuple<i64, i64>>> active_job_tasks_;
  // Track assignment of tasks to worker for this job
  struct WorkerHistory {
    timepoint_t start_time;
    timepoint_t end_time;
    i64 tasks_assigned;
    i64 tasks_retired;
  };
  std::map<i64, WorkerHistory> worker_histories_;
  std::map<i32, bool> unfinished_workers_;

  std::map<i64, std::map<i64, i64>> job_task_num_rows_;

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
