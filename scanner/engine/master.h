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

  grpc::Status Shutdown(grpc::ServerContext* context, const proto::Empty* empty,
                        Result* result);

  // Database query methods
  grpc::Status ListTables(grpc::ServerContext* context,
                         const proto::Empty* empty,
                         proto::ListTablesResult* result);

  grpc::Status GetTables(grpc::ServerContext* context,
                         const proto::GetTablesParams* params,
                         proto::GetTablesResult* result);

  grpc::Status DeleteTables(grpc::ServerContext* context,
                            const proto::DeleteTablesParams* params,
                            proto::Empty* empty);

  grpc::Status NewTable(grpc::ServerContext* context,
                        const proto::NewTableParams* params,
                        proto::Empty* empty);

  grpc::Status GetVideoMetadata(grpc::ServerContext* context,
                                const proto::GetVideoMetadataParams* params,
                                proto::GetVideoMetadataResult* result);

  // Worker methods
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

  // Op and Kernel methods

  grpc::Status GetOpInfo(grpc::ServerContext* context,
                         const proto::OpInfoArgs* op_info_args,
                         proto::OpInfo* op_info);

  grpc::Status GetSourceInfo(grpc::ServerContext* context,
                         const proto::SourceInfoArgs* source_info_args,
                         proto::SourceInfo* source_info);

  grpc::Status GetEnumeratorInfo(grpc::ServerContext* context,
                                 const proto::EnumeratorInfoArgs* info_args,
                                 proto::EnumeratorInfo* info);

  grpc::Status LoadOp(grpc::ServerContext* context,
                      const proto::OpPath* op_path, Result* result);

  grpc::Status RegisterOp(grpc::ServerContext* context,
                          const proto::OpRegistration* op_registration,
                          proto::Result* result);

  grpc::Status RegisterPythonKernel(
      grpc::ServerContext* context,
      const proto::PythonKernelRegistration* python_kernel,
      proto::Result* result);

  grpc::Status GetJobStatus(grpc::ServerContext* context,
                            const proto::Empty* empty,
                            proto::JobStatus* job_status);

  grpc::Status NextWork(grpc::ServerContext* context,
                        const proto::NodeInfo* node_info,
                        proto::NewWork* new_work);

  grpc::Status FinishedWork(grpc::ServerContext* context,
                            const proto::FinishedWorkParameters* params,
                            proto::Empty* empty);

  grpc::Status FinishedJob(grpc::ServerContext* context,
                           const proto::FinishedJobParams* params,
                           proto::Empty* empty);

  grpc::Status NewJob(grpc::ServerContext* context,
                      const proto::BulkJobParameters* job_params,
                      proto::Result* job_result);

  // Misc methods
  grpc::Status Ping(grpc::ServerContext* context, const proto::Empty* empty1,
                    proto::Empty* empty2);

  grpc::Status PokeWatchdog(grpc::ServerContext* context,
                            const proto::Empty* empty, proto::Empty* result);

  //

  void start_watchdog(grpc::Server* server, bool enable_timeout,
                      i32 timeout_ms = 50000);

 private:
  void recover_and_init_database();

  void start_job_processor();

  void stop_job_processor();

  bool process_job(const proto::BulkJobParameters* job_params,
                   proto::Result* job_result);

  void start_worker_pinger();

  void stop_worker_pinger();

  void start_job_on_workers(const std::vector<i32>& worker_ids);

  void stop_job_on_worker(i32 node_id);

  void remove_worker(i32 node_id);

  void blacklist_job(i64 job_id);

  DatabaseParameters db_params_;

  std::thread pinger_thread_;
  std::atomic<bool> pinger_active_;
  // Tracks number of times the pinger has failed to reach a worker
  std::map<i64, i64> pinger_number_of_failed_pings_;

  std::thread watchdog_thread_;
  std::atomic<bool> watchdog_awake_;
  Flag trigger_shutdown_;
  storehouse::StorageBackend* storage_;
  DatabaseMetadata meta_;
  std::unique_ptr<TableMetaCache> table_metas_;
  std::vector<std::string> so_paths_;
  std::vector<proto::OpRegistration> op_registrations_;
  std::vector<proto::PythonKernelRegistration> py_kernel_registrations_;

  // Worker state
  i32 next_worker_id_ = 0;
  std::map<i32, bool> worker_active_;
  std::map<i32, std::unique_ptr<proto::Worker::Stub>> workers_;
  std::map<i32, std::string> worker_addresses_;

  std::atomic<i64> total_tasks_used_;
  i64 total_tasks_;
  std::vector<i64> tasks_used_per_job_;

  // True if the master is executing a job
  std::mutex active_mutex_;
  std::condition_variable active_cv_;
  bool active_bulk_job_ = false;
  proto::BulkJobParameters job_params_;

  // True if all work for job is done
  std::mutex finished_mutex_;
  std::condition_variable finished_cv_;
  std::atomic<bool> finished_{true};
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
  // Job -> Task -> task output rows
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

  //============================================================================
  // Assignment of tasks to workers
  //============================================================================
  // Tracks tasks assigned to worker so they can be reassigned if the worker
  // fails
  // Worker id -> (job_id, task_id)
  std::map<i64, std::set<std::tuple<i64, i64>>> active_job_tasks_;
  // (Worker id, job_id, task_id) -> start_time
  std::map<std::tuple<i64, i64, i64>, double> active_job_tasks_starts_;
  // Tracks number of times a task has been failed so that a job can be removed
  // if it is causing consistent failures
  // job_id -> task_id -> num_failures
  std::map<i64, std::map<i64, i64>> job_tasks_num_failures_;
  // Tracks the jobs that have failed too many times and should be ignored
  std::set<i64> blacklisted_jobs_;
  struct WorkerHistory {
    timepoint_t start_time;
    timepoint_t end_time;
    i64 tasks_assigned;
    i64 tasks_retired;
  };
  std::map<i64, WorkerHistory> worker_histories_;
  std::map<i32, bool> unfinished_workers_;
  std::vector<i32> unstarted_workers_;
  std::atomic<i64> num_failed_workers_{0};
  std::vector<i32> job_uncommitted_tables_;

  // Worker connections
  std::map<std::string, i32> local_ids_;
  std::map<std::string, i32> local_totals_;
};
}
}
