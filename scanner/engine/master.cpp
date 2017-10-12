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

#include "scanner/engine/master.h"
#include "scanner/engine/ingest.h"
#include "scanner/engine/sampler.h"
#include "scanner/engine/dag_analysis.h"
#include "scanner/util/cuda.h"
#include "scanner/util/progress_bar.h"
#include "scanner/util/util.h"
#include "scanner/util/glog.h"
#include "scanner/engine/python_kernel.h"

#include <grpc/support/log.h>
#include <set>
#include <mutex>

namespace scanner {
namespace internal {

MasterImpl::MasterImpl(DatabaseParameters& params)
  : watchdog_awake_(true), db_params_(params), bar_(nullptr) {
  init_glog("scanner_master");
  storage_ =
      storehouse::StorageBackend::make_from_config(db_params_.storage_config);
  set_database_path(params.db_path);

  start_job_processor();
}

MasterImpl::~MasterImpl() {
  trigger_shutdown_.set();
  {
    std::unique_lock<std::mutex> lock(finished_mutex_);
    finished_ = true;
  }
  finished_cv_.notify_one();

  stop_job_processor();

  stop_worker_pinger();
  if (watchdog_thread_.joinable()) {
    watchdog_thread_.join();
  }
  delete storage_;
  cq_.Shutdown();
}

// Expects context->peer() to return a string in the format
// ipv4:<peer_address>:<random_port>
// Returns the <peer_address> from the above format.
std::string MasterImpl::get_worker_address_from_grpc_context(
    grpc::ServerContext* context) {
  std::string worker_address = context->peer();
  std::size_t portSep = worker_address.find_last_of(':');
  if (portSep == std::string::npos) {
  }
  std::string worker_address_base = worker_address.substr(0, portSep);

  portSep = worker_address_base.find_first_of(':');
  if (portSep == std::string::npos) {
  }

  std::string worker_address_actual = worker_address_base.substr(portSep + 1);

  return worker_address_actual;
}

grpc::Status MasterImpl::RegisterWorker(grpc::ServerContext* context,
                                        const proto::WorkerParams* worker_info,
                                        proto::Registration* registration) {
  std::unique_lock<std::mutex> lk(work_mutex_);

  set_database_path(db_params_.db_path);

  std::string worker_address = get_worker_address_from_grpc_context(context);
  worker_address += ":" + worker_info->port();

  i32 node_id = next_worker_id_++;
  VLOG(1) << "Adding worker: " << node_id << ", " << worker_address;
  workers_[node_id] = proto::Worker::NewStub(
      grpc::CreateChannel(worker_address, grpc::InsecureChannelCredentials()));
  registration->set_node_id(node_id);
  worker_addresses_[node_id] = worker_address;
  worker_active_[node_id] = true;

  // Load ops into worker
  for (const std::string& so_path : so_paths_) {
    grpc::ClientContext ctx;
    proto::OpPath op_path;
    proto::Empty empty;
    op_path.set_path(so_path);
    grpc::Status status = workers_[node_id]->LoadOp(&ctx, op_path, &empty);
    LOG_IF(FATAL, !status.ok())
        << "Master could not load op for worker at " << worker_address << " ("
        << status.error_code() << "): " << status.error_message();
  }

  if (!finished_) {
    // Update locals
    std::vector<std::string> split_addr = split(worker_address, ':');
    std::string sans_port = split_addr[0];
    if (local_totals_.count(sans_port) == 0) {
      local_totals_[sans_port] = 0;
    }
    local_totals_[sans_port] += 1;

    start_job_on_worker(node_id, worker_address);
  }

  return grpc::Status::OK;
}

grpc::Status MasterImpl::UnregisterWorker(grpc::ServerContext* context,
                                          const proto::NodeInfo* node_info,
                                          proto::Empty* empty) {
  std::unique_lock<std::mutex> lk(work_mutex_);

  set_database_path(db_params_.db_path);

  i32 node_id = node_info->node_id();
  remove_worker(node_id);

  return grpc::Status::OK;
}

grpc::Status MasterImpl::ActiveWorkers(
    grpc::ServerContext* context, const proto::Empty* empty,
    proto::RegisteredWorkers* registered_workers) {
  std::unique_lock<std::mutex> lk(work_mutex_);

  set_database_path(db_params_.db_path);

  for (auto& kv : worker_active_) {
    if (kv.second) {
      i32 worker_id = kv.first;
      proto::WorkerInfo* info = registered_workers->add_workers();
      info->set_id(worker_id);
      info->set_address(worker_addresses_.at(worker_id));
    }
  }

  return grpc::Status::OK;
}

grpc::Status MasterImpl::IngestVideos(grpc::ServerContext* context,
                                      const proto::IngestParameters* params,
                                      proto::IngestResult* result) {
  std::vector<FailedVideo> failed_videos;
  result->mutable_result()->CopyFrom(
      ingest_videos(db_params_.storage_config, db_params_.db_path,
                    std::vector<std::string>(params->table_names().begin(),
                                             params->table_names().end()),
                    std::vector<std::string>(params->video_paths().begin(),
                                             params->video_paths().end()),
                    failed_videos));
  for (auto& failed : failed_videos) {
    result->add_failed_paths(failed.path);
    result->add_failed_messages(failed.message);
  }
  return grpc::Status::OK;
}

grpc::Status MasterImpl::NextWork(grpc::ServerContext* context,
                                  const proto::NodeInfo* node_info,
                                  proto::NewWork* new_work) {
  std::unique_lock<std::mutex> lk(work_mutex_);
  VLOG(1) << "Master received NextWork command";
  if (!worker_active_.at(node_info->node_id())) {
    // Worker is not active
    new_work->set_no_more_work(true);
    return grpc::Status::OK;
  }

  // If we do not have any outstanding work, try and create more
  if (unallocated_job_tasks_.empty()) {
    // If we have no more samples for this task, try and get another task
    if (next_task_ == num_tasks_) {
      // Check if there are any tasks left
      if (next_job_ < num_jobs_ && task_result_.success()) {
        next_task_ = 0;
        num_tasks_ = job_tasks_.at(next_job_).size();
        next_job_++;
        VLOG(1) << "Jobs left: " << num_jobs_ - next_job_;
      }
    }

    // Create more work if possible
    if (next_task_ < num_tasks_) {
      i64 current_job = next_job_ - 1;
      i64 current_task = next_task_;

      unallocated_job_tasks_.push_front(
          std::make_tuple(current_job, current_task));
      next_task_++;
    }
  }

  if (unallocated_job_tasks_.empty()) {
    // No more work
    new_work->set_no_more_work(true);
    return grpc::Status::OK;
  }

  // Grab the next task sample
  std::tuple<i64, i64> job_task_id = unallocated_job_tasks_.back();
  unallocated_job_tasks_.pop_back();

  assert(next_task_ <= num_tasks_);

  i64 job_idx;
  i64 task_idx;
  std::tie(job_idx, task_idx) = job_task_id;
  new_work->set_table_id(job_to_table_id_.at(job_idx));
  new_work->set_job_index(job_idx);
  new_work->set_task_index(task_idx);
  const auto& task_rows = job_tasks_.at(job_idx).at(task_idx);
  for (i64 r : task_rows) {
    new_work->add_output_rows(r);
  }

  // Track sample assigned to worker
  active_job_tasks_[node_info->node_id()].insert(job_task_id);
  worker_histories_[node_info->node_id()].tasks_assigned += 1;

  return grpc::Status::OK;
}

grpc::Status MasterImpl::FinishedWork(
    grpc::ServerContext* context, const proto::FinishedWorkParameters* params,
    proto::Empty* empty) {
  std::unique_lock<std::mutex> lk(work_mutex_);
  VLOG(1) << "Master received FinishedWork command";

  i32 worker_id = params->node_id();
  i64 job_id = params->job_id();
  i64 task_id = params->task_id();
  i64 num_rows = params->num_rows();

  if (!worker_active_[worker_id]) {
    // Technically the task was finished, but we don't count it for now
    // because it would have been reinstered into the work queue
    return grpc::Status::OK;
  }

  auto& worker_tasks = active_job_tasks_.at(worker_id);

  std::tuple<i64, i64> job_tasks = std::make_tuple(job_id, task_id);
  assert(worker_tasks.count(job_tasks) > 0);
  worker_tasks.erase(job_tasks);

  worker_histories_[worker_id].tasks_retired += 1;

  i64 active_job = next_job_ - 1;

  total_tasks_used_++;
  if (bar_) {
    bar_->Progressed(total_tasks_used_);
  }

  if (total_tasks_used_ == total_tasks_) {
    VLOG(1) << "Master FinishedWork triggered finished!";
    assert(next_job_ == num_jobs_);
    {
      std::unique_lock<std::mutex> lock(finished_mutex_);
      finished_ = true;
    }
    finished_cv_.notify_all();
  }

  return grpc::Status::OK;
}

grpc::Status MasterImpl::NewJob(grpc::ServerContext* context,
                                const proto::BulkJobParameters* job_params,
                                proto::Result* job_result) {
  VLOG(1) << "Master received NewJob command";
  job_result->set_success(true);
  set_database_path(db_params_.db_path);

  job_params_.Clear();
  job_params_.MergeFrom(*job_params);
  {
    std::unique_lock<std::mutex> lock(finished_mutex_);
    finished_ = false;
  }
  finished_cv_.notify_one();

  {
    std::unique_lock<std::mutex> lock(active_mutex_);
    active_bulk_job_ = true;
  }
  active_cv_.notify_all();

  return grpc::Status::OK;
}

grpc::Status MasterImpl::IsJobDone(grpc::ServerContext* context,
                                   const proto::Empty* empty,
                                   proto::JobResult* job_result) {
  VLOG(1) << "Master received IsJobDone command";
  std::unique_lock<std::mutex> lock(active_mutex_);
  if (!active_bulk_job_) {
    job_result->set_finished(true);
    job_result->mutable_result()->CopyFrom(job_result_);
  } else {
    job_result->set_finished(false);
  }
  return grpc::Status::OK;
}


grpc::Status MasterImpl::Ping(grpc::ServerContext* context,
                              const proto::Empty* empty1,
                              proto::Empty* empty2) {
  return grpc::Status::OK;
}

grpc::Status MasterImpl::GetOpInfo(grpc::ServerContext* context,
                                   const proto::OpInfoArgs* op_info_args,
                                   proto::OpInfo* op_info) {
  OpRegistry* registry = get_op_registry();
  std::string op_name = op_info_args->op_name();
  if (!registry->has_op(op_name)) {
    op_info->mutable_result()->set_success(false);
    op_info->mutable_result()->set_msg("Op " + op_name + " does not exist");
    return grpc::Status::OK;
  }

  OpInfo* info = registry->get_op_info(op_name);

  op_info->set_variadic_inputs(info->variadic_inputs());
  for (auto& input_column : info->input_columns()) {
    Column* info = op_info->add_input_columns();
    info->CopyFrom(input_column);
  }
  for (auto& output_column : info->output_columns()) {
    Column* info = op_info->add_output_columns();
    info->CopyFrom(output_column);
  }
  op_info->mutable_result()->set_success(true);

  return grpc::Status::OK;
}

grpc::Status MasterImpl::LoadOp(grpc::ServerContext* context,
                                const proto::OpPath* op_path, Result* result) {
  std::unique_lock<std::mutex> lk(work_mutex_);
  const std::string& so_path = op_path->path();
  {
    std::ifstream infile(so_path);
    if (!infile.good()) {
      RESULT_ERROR(result, "Op library was not found: %s", so_path.c_str());
      return grpc::Status::OK;
    }
  }

  void* handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    RESULT_ERROR(result, "Failed to load op library: %s", dlerror());
    return grpc::Status::OK;
  }
  so_paths_.push_back(so_path);

  for (auto& kv : worker_active_) {
    if (kv.second) {
      auto& worker = workers_[kv.first];
      grpc::ClientContext ctx;
      proto::Empty empty;
      worker->LoadOp(&ctx, *op_path, &empty);
    }
  }

  result->set_success(true);
  return grpc::Status::OK;
}

grpc::Status MasterImpl::RegisterOp(
    grpc::ServerContext* context, const proto::OpRegistration* op_registration,
    proto::Result* result) {
  std::unique_lock<std::mutex> lk(work_mutex_);
  VLOG(1) << "Master registering Op: " << op_registration->name();

  result->set_success(true);
  const std::string& name = op_registration->name();
  {
    const bool variadic_inputs = op_registration->variadic_inputs();
    std::vector<Column> input_columns;
    size_t i = 0;
    for (auto& c : op_registration->input_columns()) {
      Column col;
      col.set_id(i++);
      col.set_name(c.name());
      col.set_type(c.type());
      input_columns.push_back(col);
    }
    std::vector<Column> output_columns;
    i = 0;
    for (auto& c : op_registration->output_columns()) {
      Column col;
      col.set_id(i++);
      col.set_name(c.name());
      col.set_type(c.type());
      output_columns.push_back(col);
    }
    bool can_stencil = op_registration->can_stencil();
    std::vector<i32> stencil(op_registration->preferred_stencil().begin(),
                             op_registration->preferred_stencil().end());
    if (stencil.empty()) {
      stencil = {0};
    }
    bool has_bounded_state = op_registration->has_bounded_state();
    i32 warmup = op_registration->warmup();
    bool has_unbounded_state = op_registration->has_unbounded_state();
    OpInfo* info = new OpInfo(name, variadic_inputs, input_columns,
                              output_columns, can_stencil, stencil,
                              has_bounded_state, warmup, has_unbounded_state);
    OpRegistry* registry = get_op_registry();
    *result = registry->add_op(name, info);
  }
  if (!result->success()) {
    LOG(WARNING) << "Master failed to register op " << name;
    return grpc::Status::OK;
  }

  for (auto& kv : worker_active_) {
    if (kv.second) {
      auto& worker = workers_[kv.first];
      grpc::ClientContext ctx;
      proto::Result w_result;
      worker->RegisterOp(&ctx, *op_registration, &w_result);
    }
  }

  op_registrations_.push_back(*op_registration);
  return grpc::Status::OK;
}

grpc::Status MasterImpl::RegisterPythonKernel(
    grpc::ServerContext* context,
    const proto::PythonKernelRegistration* python_kernel,
    proto::Result* result) {
  std::unique_lock<std::mutex> lk(work_mutex_);
  VLOG(1) << "Master registering Python Kernel: " << python_kernel->op_name();

  {
    const std::string& op_name = python_kernel->op_name();
    DeviceType device_type = python_kernel->device_type();
    const std::string& kernel_str = python_kernel->kernel_str();
    const std::string& pickled_config = python_kernel->pickled_config();
    // Create a kernel builder function
    auto constructor = [kernel_str, pickled_config](const KernelConfig& config) {
      return new PythonKernel(config, kernel_str, pickled_config);
    };
    // Create a new kernel factory
    // TODO(apoms): Support batching and # of devices in python kernels
    KernelFactory* factory =
        new KernelFactory(op_name, device_type, 1, false, 1, constructor);
    // Register the kernel
    KernelRegistry* registry = get_kernel_registry();
    registry->add_kernel(op_name, factory);
  }

  for (auto& kv : worker_active_) {
    if (kv.second) {
      auto& worker = workers_[kv.first];
      grpc::ClientContext ctx;
      proto::Result w_result;
      worker->RegisterPythonKernel(&ctx, *python_kernel, &w_result);
    }
  }

  py_kernel_registrations_.push_back(*python_kernel);
  result->set_success(true);
  return grpc::Status::OK;
}


grpc::Status MasterImpl::Shutdown(grpc::ServerContext* context,
                                  const proto::Empty* empty, Result* result) {
  VLOG(1) << "Master received shutdown!";
  result->set_success(true);
  trigger_shutdown_.set();
  return grpc::Status::OK;
}

grpc::Status MasterImpl::PokeWatchdog(grpc::ServerContext* context,
                                      const proto::Empty* empty,
                                      proto::Empty* result) {
  watchdog_awake_ = true;

  std::map<i32, proto::Worker::Stub*> ws;
  {
    std::unique_lock<std::mutex> lk(work_mutex_);
    for (auto& kv : workers_) {
      i32 worker_id = kv.first;
      auto& worker = kv.second;
      if (!worker_active_[worker_id]) continue;

      ws.insert({worker_id, kv.second.get()});
    }
  }

  std::vector<grpc::ClientContext> contexts(ws.size());
  std::vector<grpc::Status> statuses(ws.size());
  std::vector<proto::Empty> results(ws.size());
  std::vector<std::unique_ptr<grpc::ClientAsyncResponseReader<proto::Empty>>>
      rpcs(ws.size());
  grpc::CompletionQueue cq;
  int i = 0;
  for (auto& kv : ws) {
    i64 id = kv.first;
    auto& worker = kv.second;
    proto::Empty em;
    rpcs[i] = worker->AsyncPokeWatchdog(&contexts[i], em, &cq);
    rpcs[i]->Finish(&results[i], &statuses[i], (void*)id);
    i++;
  }
  for (int i = 0; i < ws.size(); ++i) {
    void* got_tag;
    bool ok = false;
    GPR_ASSERT(cq.Next(&got_tag, &ok));
    // GPR_ASSERT((i64)got_tag < workers_.size());
    i64 worker_id = (i64)got_tag;
    if (!ok) {
      LOG(WARNING) << "Could not poke worker " << worker_id << "!";
    }
  }
  cq.Shutdown();
  return grpc::Status::OK;
}

void MasterImpl::start_watchdog(grpc::Server* server, bool enable_timeout,
                                i32 timeout_ms) {
  watchdog_thread_ = std::thread([this, server, enable_timeout, timeout_ms]() {
    double time_since_check = 0;
    // Wait until shutdown is triggered or watchdog isn't woken up
    if (!enable_timeout) {
      trigger_shutdown_.wait();
    }
    while (!trigger_shutdown_.raised()) {
      auto sleep_start = now();
      trigger_shutdown_.wait_for(timeout_ms);
      time_since_check += nano_since(sleep_start) / 1e6;
      if (time_since_check > timeout_ms) {
        if (!watchdog_awake_) {
          // Watchdog not woken, time to bail out
          LOG(ERROR) << "Master did not receive heartbeat in " << timeout_ms
                     << "ms. Shutting down.";
          trigger_shutdown_.set();
        }
        watchdog_awake_ = false;
        time_since_check = 0;
      }
    }
    // Shutdown workers
    std::vector<i32> worker_ids;
    {
      std::unique_lock<std::mutex> lk(work_mutex_);
      for (auto& kv : workers_) {
        worker_ids.push_back(kv.first);
      }
    }
    for (i32 i : worker_ids) {
      grpc::ClientContext ctx;
      proto::Empty empty;
      proto::Result wresult;
      workers_.at(i)->Shutdown(&ctx, empty, &wresult);
    }
    // Shutdown self
    server->Shutdown();
  });
}

void MasterImpl::start_job_processor() {
  job_processor_thread_ = std::thread([this]() {
    while (!trigger_shutdown_.raised()) {
      // Wait on not finished
      {
        std::unique_lock<std::mutex> lock(active_mutex_);
        active_cv_.wait(
            lock, [this] { return active_bulk_job_ || trigger_shutdown_.raised(); });
      }
      if (trigger_shutdown_.raised()) break;
      // Start processing job
      bool result = process_job(&job_params_, &job_result_);
    }
  });
}

void MasterImpl::stop_job_processor() {
  // Wake up job processor
  {
    std::unique_lock<std::mutex> lock(active_mutex_);
    active_bulk_job_ = true;
  }
  active_cv_.notify_all();
  if (job_processor_thread_.joinable()) {
    job_processor_thread_.join();
  }
}

bool MasterImpl::process_job(const proto::BulkJobParameters* job_params,
                             proto::Result* job_result) {
  // Reset job state
  job_to_table_id_.clear();
  slice_input_rows_per_job_.clear();
  total_output_rows_per_job_.clear();
  unallocated_job_tasks_.clear();
  next_job_ = 0;
  num_jobs_ = -1;
  next_task_ = 0;
  num_tasks_ = -1;
  task_result_.set_success(true);
  active_job_tasks_.clear();
  worker_histories_.clear();
  unfinished_workers_.clear();
  local_ids_.clear();
  local_totals_.clear();
  client_contexts_.clear();
  statuses_.clear();
  replies_.clear();
  rpcs_.clear();
  total_tasks_used_ = 0;
  total_tasks_ = 0;

  job_result->set_success(true);

  auto finished_fn = [this]() {
    {
      std::unique_lock<std::mutex> lock(finished_mutex_);
      finished_ = true;
    }
    finished_cv_.notify_all();
    {
      std::unique_lock<std::mutex> lock(finished_mutex_);
      active_bulk_job_ = false;
    }
    active_cv_.notify_all();
  };

  std::vector<proto::Job> jobs(job_params->jobs().begin(),
                               job_params->jobs().end());
  std::vector<proto::Op> ops(job_params->ops().begin(),
                             job_params->ops().end());

  const i32 io_packet_size = job_params->io_packet_size();
  const i32 work_packet_size = job_params->work_packet_size();
  if (io_packet_size > 0 && io_packet_size % work_packet_size != 0) {
    RESULT_ERROR(job_result,
                 "IO packet size must be a multiple of Work packet size.");
    finished_fn();
    return false;
  }

  i32 warmup_size = 0;
  i32 total_rows = 0;

  meta_ = read_database_metadata(storage_, DatabaseMetadata::descriptor_path());
  DatabaseMetadata meta_copy =
      read_database_metadata(storage_, DatabaseMetadata::descriptor_path());

  // Setup table metadata cache
  table_metas_.reset(new TableMetaCache(storage_, meta_));

  // Prefetch table metadata for all tables in samplers
  {
    auto load_table_meta = [&](const std::string& table_name) {
      std::string table_path =
          TableMetadata::descriptor_path(meta_.get_table_id(table_name));
      table_metas_->update(read_table_metadata(storage_, table_path));
    };
    std::set<std::string> tables_to_read;
    for (auto& job : jobs) {
      for (auto& column_input : job.inputs()) {
        tables_to_read.insert(column_input.table_name());
      }
    }
    // TODO(apoms): make this a thread pool instead of spawning potentially
    // thousands of threads
    std::vector<std::thread> threads;
    for (const std::string& t : tables_to_read) {
      threads.emplace_back(load_table_meta, t);
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }

  DAGAnalysisInfo dag_info;
  *job_result =
      validate_jobs_and_ops(meta_, *table_metas_.get(), jobs, ops, dag_info);
  if (!job_result->success()) {
    // No database changes made at this point, so just return
    finished_fn();
    return false;
  }

  // Map all input Ops into a single input collection
  const std::map<i64, i64>& input_op_idx_to_column_idx = dag_info.input_ops;

  // Get output columns from last output op to set as output table columns
  OpRegistry* op_registry = get_op_registry();
  auto& last_op = ops.at(ops.size() - 1);
  assert(last_op.name() == OUTPUT_OP_NAME);
  std::vector<std::vector<Column>> job_output_columns;
  for (const auto& job : jobs) {
    // Get input columns from column inputs specified for each job
    std::map<i64, Column> input_op_idx_to_column;
    {
      for (auto& ci : job.inputs()) {
        const TableMetadata& table = table_metas_->at(ci.table_name());
        std::vector<Column> table_columns = table.columns();
        const std::string& c = ci.column_name();
        bool found = false;
        for (Column& col : table_columns) {
          if (c == col.name()) {
            Column new_col;
            new_col.CopyFrom(col);
            new_col.set_id(0);
            input_op_idx_to_column[ci.op_index()] = new_col;
            found = true;
            break;
          }
        }
        assert(found);
      }
    }

    job_output_columns.emplace_back();
    std::vector<Column>& output_columns = job_output_columns.back();
    for (const auto& input : last_op.inputs()) {
      auto& input_op = ops.at(input.op_index());
      std::vector<Column> input_columns;
      if (input_op.name() == INPUT_OP_NAME) {
        input_columns = {input_op_idx_to_column.at(input.op_index())};
      } else {
        OpInfo* input_op_info = op_registry->get_op_info(input_op.name());
        input_columns = input_op_info->output_columns();
      }
      const std::string& name = input.column();
      bool found = false;
      for (auto& col : input_columns) {
        if (col.name() == name) {
          Column c;
          c.set_id(output_columns.size());
          c.set_name(name);
          c.set_type(col.type());
          output_columns.push_back(c);
          found = true;
          break;
        }
      }
      assert(found);
    }
  }
  proto::BulkJobDescriptor job_descriptor;
  job_descriptor.set_io_packet_size(io_packet_size);
  job_descriptor.set_work_packet_size(work_packet_size);
  job_descriptor.set_num_nodes(workers_.size());

  {
    auto& jobs = job_params->jobs();
    job_descriptor.mutable_jobs()->CopyFrom(jobs);
  }

  // Add job name into database metadata so we can look up what jobs have
  // been run
  i32 bulk_job_id = meta_.add_bulk_job(job_params->job_name());
  job_descriptor.set_id(bulk_job_id);
  job_descriptor.set_name(job_params->job_name());

  if (!job_result->success()) {
    // No database changes made at this point, so just return
    finished_fn();
    return false;
  }

  // Determine total output rows and slice input rows for using to
  // split stream
  *job_result = determine_input_rows_to_slices(meta_, *table_metas_.get(), jobs,
                                               ops, dag_info);
  slice_input_rows_per_job_ = dag_info.slice_input_rows;
  total_output_rows_per_job_ = dag_info.total_output_rows;

  // HACK(apoms): we currently split work into tasks in two ways:
  //  a) align with the natural boundaries defined by the slice partitioner
  //  b) use a user-specified size to chunk up the output sequence

  // Job -> task -> rows
  total_tasks_ = 0;
  for (size_t i = 0; i < jobs.size(); ++i) {
    auto& slice_input_rows = slice_input_rows_per_job_[i];
    i64 total_output_rows = total_output_rows_per_job_[i];

    std::vector<i64> partition_boundaries;
    if (slice_input_rows.size() == 0) {
      // No slices, so we can split as desired. Currently use IO packet size
      // since it is the smallest granularity we can specify
      for (i64 r = 0; r < total_output_rows;
           r += job_params->io_packet_size()) {
        partition_boundaries.push_back(r);
      }
      partition_boundaries.push_back(total_output_rows);
    } else {
      // Split stream into partitions, respecting slice boundaries
      // We assume there is only one slice for now since
      // they all must have the same number of groups
      assert(slice_input_rows.size() == 1);
      // Derive the output rows produced by each slice group
      i64 slice_op_idx = slice_input_rows.begin()->first;
      i64 slice_in_rows = slice_input_rows.begin()->second;
      *job_result = derive_slice_final_output_rows(
          jobs.at(i), ops, slice_op_idx, slice_in_rows, dag_info,
          partition_boundaries);
      if (!job_result->success()) {
        // No database changes made at this point, so just return
        finished_fn();
        return false;
      }
    }
    assert(partition_boundaries.back() == total_output_rows);
    job_tasks_.emplace_back();
    auto& tasks = job_tasks_.back();
    for (i64 pi = 0; pi < partition_boundaries.size() - 1; ++pi) {
      tasks.emplace_back();
      auto& task_rows = tasks.back();

      i64 s = partition_boundaries[pi];
      i64 e = partition_boundaries[pi + 1];
      for (i64 r = s; r < e; ++r) {
        task_rows.push_back(r);
      }
      total_tasks_++;
    }
  }

  if (!job_result->success()) {
    // No database changes made at this point, so just return
    finished_fn();
    return false;
  }

  // Write out database metadata so that workers can read it
  write_bulk_job_metadata(storage_, BulkJobMetadata(job_descriptor));

  for (i64 job_idx = 0; job_idx < job_params->jobs_size(); ++job_idx) {
    auto& job = job_params->jobs(job_idx);
    i32 table_id = meta_.add_table(job.output_table_name());
    job_to_table_id_[job_idx] = table_id;
    proto::TableDescriptor table_desc;
    table_desc.set_id(table_id);
    table_desc.set_name(job.output_table_name());
    table_desc.set_timestamp(std::chrono::duration_cast<std::chrono::seconds>(
                                 now().time_since_epoch())
                                 .count());
    // Set columns equal to the last op's output columns
    for (size_t i = 0; i < job_output_columns[job_idx].size(); ++i) {
      Column* col = table_desc.add_columns();
      col->CopyFrom(job_output_columns[job_idx][i]);
    }
    table_metas_->update(TableMetadata(table_desc));

    i64 total_rows = 0;
    std::vector<i64> end_rows;
    auto& tasks = job_tasks_.at(job_idx);
    for (i64 task_id = 0; task_id < tasks.size(); ++task_id) {
      i64 task_rows = tasks.at(task_id).size();
      total_rows += task_rows;
      end_rows.push_back(total_rows);
    }
    for (i64 r : end_rows) {
      table_desc.add_end_rows(r);
    }
    table_desc.set_job_id(job_idx);

    write_table_metadata(storage_, TableMetadata(table_desc));
    table_metas_->update(TableMetadata(table_desc));
  }

  // Setup initial task sampler
  task_result_.set_success(true);
  next_task_ = 0;
  num_tasks_ = 0;
  next_job_ = 0;
  num_jobs_ = jobs.size();

  write_database_metadata(storage_, meta_);

  VLOG(1) << "Total jobs: " << num_jobs_;

  if (job_params->show_progress()) {
    bar_.reset(new ProgressBar(total_tasks_, ""));
  } else {
    bar_.reset(nullptr);
  }

  // TODO(apoms): change this to support adding and removing nodes
  //              the main change is that the workers should handle
  //              spawning sub processes instead of appearing as
  //              multiple logical nodes
  for (auto kv : worker_addresses_) {
    const std::string& address = kv.second;
    // Strip port
    std::vector<std::string> split_addr = split(address, ':');
    std::string sans_port = split_addr[0];
    if (local_totals_.count(sans_port) == 0) {
      local_totals_[sans_port] = 0;
    }
    local_totals_[sans_port] += 1;
  }

  // Send new job command to workers
  VLOG(1) << "Sending new job command to workers";

  {
    std::unique_lock<std::mutex> lk(work_mutex_);
    for (auto kv : worker_addresses_) {
      i32 worker_id = kv.first;
      std::string& address = kv.second;

      start_job_on_worker(worker_id, address);
    }
  }

  // Ping workers every 10 seconds to make sure they are alive
  start_worker_pinger();

  // Wait for all workers to finish
  VLOG(1) << "Waiting for workers to finish";

  auto check_worker_fn = [&]() {
    void* got_tag;
    bool ok = false;
    GPR_ASSERT(cq_.Next(&got_tag, &ok));
    // GPR_ASSERT((i64)got_tag < workers_.size());
    assert(ok);

    i64 worker_id = (i64)got_tag;
    VLOG(2) << "Worker " << worker_id << " finished.";

    std::unique_lock<std::mutex> lk(work_mutex_);
    if (worker_active_[worker_id] && !replies_[worker_id]->success()) {
      LOG(WARNING) << "Worker " << worker_id
                   << " returned error: " << replies_[worker_id]->msg();
      job_result->set_success(false);
      job_result->set_msg(replies_[worker_id]->msg());
      next_job_ = num_jobs_;
    }
    unfinished_workers_[worker_id] = false;
  };
  // Wait until all workers are done and work has been completed
  while (!finished_) {
    check_worker_fn();
  }

  {
    std::unique_lock<std::mutex> lock(finished_mutex_);
    finished_cv_.wait(lock, [this] { return finished_; });
  }
  // Get responses for all active workers that we have not gotten responses
  // for yet
  i32 num_unfinished = 0;
  {
    std::unique_lock<std::mutex> lk(work_mutex_);
    for (auto& kv : unfinished_workers_) {
      i32 worker_id = kv.first;
      if (kv.second && worker_active_[worker_id]) {
        num_unfinished++;
      }
    }
  }
  for (int i = 0; i < num_unfinished; ++i) {
    check_worker_fn();
  }

  // No need to check status of workers anymore
  stop_worker_pinger();

  if (!job_result->success()) {
    // Overwrite database metadata with copy from prior to modification
    write_database_metadata(storage_, meta_copy);
  }

  if (!task_result_.success()) {
    job_result->CopyFrom(task_result_);
  } else {
    assert(next_job_ == num_jobs_);
    if (bar_) {
      bar_->Progressed(total_tasks_);
    }
  }

  std::fflush(NULL);
  sync();

  finished_fn();

  VLOG(1) << "Master finished job";
}

void MasterImpl::start_worker_pinger() {
  while (!finished_) {
    std::map<i32, proto::Worker::Stub*> ws;
    {
      std::unique_lock<std::mutex> lk(work_mutex_);
      for (auto& kv : workers_) {
        i32 worker_id = kv.first;
        auto& worker = kv.second;
        if (!worker_active_[worker_id]) continue;

        ws.insert({worker_id, kv.second.get()});
      }
    }

    for (auto& kv : ws) {
      i32 worker_id = kv.first;
      auto& worker = kv.second;

      grpc::ClientContext ctx;
      proto::Empty empty1;
      proto::Empty empty2;
      grpc::Status status = worker->Ping(&ctx, empty1, &empty2);
      if (!status.ok()) {
        // Worker not responding, remove it from active workers
        LOG(WARNING) << "Worker " << worker_id << " did not respond to Ping. "
                     << "Removing worker from active list.";
        remove_worker(worker_id);
      }
    }
    // FIXME(apoms): this sleep is unfortunate because it means a
    //               job must take at least this long. A solution
    //               would be to put it in a separate thread.
    std::this_thread::sleep_for(std::chrono::seconds(5));
  }
}

void MasterImpl::stop_worker_pinger() {
}

void MasterImpl::start_job_on_worker(i32 worker_id,
                                     const std::string& address) {
  proto::BulkJobParameters w_job_params;
  w_job_params.MergeFrom(job_params_);

  auto& worker = workers_.at(worker_id);
  std::vector<std::string> split_addr = split(address, ':');
  std::string sans_port = split_addr[0];
  w_job_params.set_local_id(local_ids_[sans_port]);
  w_job_params.set_local_total(local_totals_[sans_port]);
  local_ids_[sans_port] += 1;
  client_contexts_[worker_id] =
      std::unique_ptr<grpc::ClientContext>(new grpc::ClientContext);
  statuses_[worker_id] = std::unique_ptr<grpc::Status>(new grpc::Status);
  replies_[worker_id] = std::unique_ptr<proto::Result>(new proto::Result);
  rpcs_[worker_id] = worker->AsyncNewJob(client_contexts_[worker_id].get(),
                                         w_job_params, &cq_);
  rpcs_[worker_id]->Finish(replies_[worker_id].get(),
                           statuses_[worker_id].get(), (void*)worker_id);
  worker_histories_[worker_id].start_time = now();
  worker_histories_[worker_id].tasks_assigned = 0;
  worker_histories_[worker_id].tasks_retired = 0;
  unfinished_workers_[worker_id] = true;
  VLOG(2) << "Sent NewJob command to worker " << worker_id;
}

void MasterImpl::stop_job_on_worker(i32 worker_id) {
  // Place workers active tasks back into the unallocated task samples
  if (active_job_tasks_.count(worker_id) > 0) {
    // Place workers active tasks back into the unallocated task samples
    for (const std::tuple<i64, i64>& worker_job_task :
         active_job_tasks_.at(worker_id)) {
      unallocated_job_tasks_.push_back(worker_job_task);
    }
    VLOG(1) << "Reassigning worker " << worker_id << "'s "
            << active_job_tasks_.at(worker_id).size() << " task samples.";
    active_job_tasks_.erase(worker_id);
  }

  worker_histories_[worker_id].end_time = now();
  unfinished_workers_[worker_id] = false;

  // Remove async job command data
  assert(client_contexts_.count(worker_id) > 0);
  client_contexts_[worker_id]->TryCancel();
  /*client_contexts_.erase(worker_id);
  statuses_.erase(worker_id);
  replies_.erase(worker_id);
  rpcs_.erase(worker_id);*/
}

void MasterImpl::remove_worker(i32 node_id) {
  assert(workers_.count(node_id) > 0);

  std::string worker_address = worker_addresses_.at(node_id);
  // Remove worker from list
  worker_active_[node_id] = false;

  {
    std::unique_lock<std::mutex> lock(active_mutex_);
    if (active_bulk_job_ && client_contexts_.count(node_id) > 0) {
      stop_job_on_worker(node_id);
    }
  }

  // Update locals
  /*std::vector<std::string> split_addr = split(worker_address, ':');
  std::string sans_port = split_addr[0];
  assert(local_totals_.count(sans_port) > 0);
  local_totals_[sans_port] -= 1;
  local_ids_[sans_port] -= 1;*/

  VLOG(1) << "Removing worker " << node_id << " (" << worker_address << ").";

}

}
}
