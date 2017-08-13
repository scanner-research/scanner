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
#include <grpc/support/log.h>
#include <mutex>
#include "scanner/engine/ingest.h"
#include "scanner/engine/sampler.h"
#include "scanner/util/cuda.h"
#include "scanner/util/progress_bar.h"
#include "scanner/util/util.h"
#include "scanner/util/glog.h"

namespace scanner {
namespace internal {
namespace {
void validate_task_set(DatabaseMetadata& meta, const proto::TaskSet& task_set,
                       Result* result) {
  auto& tasks = task_set.tasks();
  // Validate tasks
  std::set<std::string> task_output_table_names;
  for (auto& task : task_set.tasks()) {
    if (task.output_table_name() == "") {
      LOG(WARNING) << "Task specified with empty output table name. Output "
                      "tables can not have empty names";
      result->set_success(false);
    }
    if (meta.has_table(task.output_table_name())) {
      LOG(WARNING) << "Task specified with duplicate output table name. "
                   << "A table with name " << task.output_table_name() << " "
                   << "already exists.";
      result->set_success(false);
    }
    if (task_output_table_names.count(task.output_table_name()) > 0) {
      LOG(WARNING) << "Multiple tasks specified with output table name "
                   << task.output_table_name()
                   << ". Table names must be unique.";
      result->set_success(false);
    }
    task_output_table_names.insert(task.output_table_name());
    if (task.samples().size() == 0) {
      LOG(WARNING) << "Task " << task.output_table_name() << " did not "
                   << "specify any tables to sample from. Tasks must sample "
                   << "from at least one table.";
      result->set_success(false);
    } else {
      for (auto& sample : task.samples()) {
        if (!meta.has_table(sample.table_name())) {
          LOG(WARNING) << "Task " << task.output_table_name() << " tried to "
                       << "sample from non-existent table "
                       << sample.table_name()
                       << ". TableSample must sample from existing table.";
          result->set_success(false);
        }
        // TODO(apoms): validate sampler functions
        if (sample.column_names().size() == 0) {
          LOG(WARNING) << "Task" << task.output_table_name() << " tried to "
                       << "sample zero columns from table "
                       << sample.table_name()
                       << ". TableSample must sample at least one column";
          result->set_success(false);
        }
      }
    }
  }
  // Validate ops
  {
    OpRegistry* op_registry = get_op_registry();
    KernelRegistry* kernel_registry = get_kernel_registry();

    i32 op_idx = 0;
    std::vector<std::string> op_names;
    std::vector<std::vector<std::string>> op_outputs;
    for (auto& op : task_set.ops()) {
      op_names.push_back(op.name());

      if (op_idx == 0) {
        if (op.name() != "InputTable") {
          RESULT_ERROR(result, "First Op is %s but must be Op InputTable",
                       op.name().c_str());
          break;
        }
        op_outputs.emplace_back();
        for (auto& input : op.inputs()) {
          for (auto& col : input.columns()) {
            op_outputs.back().push_back(col);
          }
        }
        op_idx++;
        continue;
      }
      if (op.name() != "OutputTable") {
        op_outputs.emplace_back();
        if (!op_registry->has_op(op.name())) {
          RESULT_ERROR(result, "Op %s is not registered.", op.name().c_str());
        } else {
          for (auto& col :
               op_registry->get_op_info(op.name())->output_columns()) {
            op_outputs.back().push_back(col.name());
          }
        }
        if (!kernel_registry->has_kernel(op.name(), op.device_type())) {
          RESULT_ERROR(result,
                       "Op %s at index %d requested kernel with device type "
                       "%s but no such kernel exists.",
                       op.name().c_str(), op_idx,
                       (op.device_type() == DeviceType::CPU ? "CPU" : "GPU"));
        }
      }
      i32 input_count = 0;
      for (auto& input : op.inputs()) {
        if (input.op_index() >= op_idx) {
          RESULT_ERROR(result,
                       "Op %s at index %d referenced input index %d."
                       "Ops must be specified in topo sort order.",
                       op.name().c_str(), op_idx, input.op_index());
        } else {
          std::string& input_op_name = op_names.at(input.op_index());
          std::vector<std::string>& inputs = op_outputs.at(input.op_index());
          input_count += input.columns().size();
          for (auto& col : input.columns()) {
            bool found = false;
            for (auto& out_col : inputs) {
              if (col == out_col) {
                found = true;
                break;
              }
            }
            if (!found) {
              RESULT_ERROR(result,
                           "Op %s at index %d requested column %s from input "
                           "Op %s at index %d but that Op does not have the "
                           "requsted column.",
                           op.name().c_str(), op_idx, col.c_str(),
                           input_op_name.c_str(), input.op_index());
            }
          }
        }
      }
      if (op.name() != "OutputTable") {
        OpInfo* info = op_registry->get_op_info(op.name());
        KernelFactory* factory =
            kernel_registry->get_kernel(op.name(), op.device_type());
        // Check that the # of inputs match up
        // TODO(apoms): type check for frame
        if (!info->variadic_inputs()) {
          i32 expected_inputs = info->input_columns().size();
          if (expected_inputs != input_count) {
            RESULT_ERROR(
                result,
                "Op %s at index %d expects %d input columns, but received %d",
                op.name().c_str(), op_idx, expected_inputs, input_count);
          }
        }

        // Check that a stencil is not set on a non-stenciling kernel
        // If can't stencil, then should have a zero size stencil or a size 1
        // stencil with the element 0
        if (!info->can_stencil() &&
            !((op.stencil_size() == 0) ||
              (op.stencil_size() == 1 && op.stencil(0) == 0))) {
          RESULT_ERROR(
              result,
              "Op %s at index %d specified stencil but that Op was not "
              "declared to support stenciling. Add .stencil() to the Op "
              "declaration to support stenciling.",
              op.name().c_str(), op_idx);
        }
        // Check that a stencil is not set on a non-stenciling kernel
        if (!factory->can_batch() && op.batch() > 1) {
          RESULT_ERROR(
              result,
              "Op %s at index %d specified a batch size but the Kernel for "
              "that Op was not declared to support batching. Add .batch() to "
              "the Kernel declaration to support batching.",
              op.name().c_str(), op_idx);
        }
      }
      op_idx++;
    }
    if (op_names.size() < 3) {
      RESULT_ERROR(result,
                   "Task set must specify at least three Ops: "
                   "an InputTable Op, any other Op, and an OutputTable Op. "
                   "However, only %lu Ops were specified.",
                   op_names.size());
    } else {
      if (op_names.front() != "InputTable") {
        RESULT_ERROR(result, "First Op is %s but must be InputTable",
                     op_names.front().c_str());
      }
      if (op_names.back() != "OutputTable") {
        RESULT_ERROR(result, "Last Op is %s but must be OutputTable",
                     op_names.back().c_str());
      }
    }
  }
}

Result get_task_end_rows(
    const TableMetaCache& table_metas,
    const proto::Task& task, i64 min_stencil, i64 max_stencil,
    std::vector<i64>& rows) {
  Result result;
  result.set_success(true);

  std::vector<i64> table_num_rows;
  for (auto& s : task.samples()) {
    table_num_rows.push_back(table_metas.at(s.table_name()).num_rows());
  }

  TaskSampler sampler(table_metas, task);
  result = sampler.validate();
  if (!result.success()) {
    return result;
  }
  i64 start_rows_lost = 0;
  i64 num_samples = sampler.total_samples();
  for (i64 i = 0; i < num_samples; ++i) {
    proto::NewWork new_work;
    result = sampler.next_work(new_work);
    if (!result.success()) {
      rows.clear();
      return result;
    }

    i64 requested_start_row = new_work.io_item().start_row();
    i64 requested_end_row = new_work.io_item().end_row();

    i64 work_item_start_reduction = 0;
    i64 work_item_end_reduction = 0;
    for (i32 j = 0; j < new_work.load_work().samples_size(); j++) {
      auto& s = new_work.load_work().samples(j);
      // If this IO item is near the start or end, we should check if it
      // is attempting to produce invalid rows due to a stencil
      // requirement that can not be fulfilled

      // Check if near start
      i64 min_requested = s.rows(0) + min_stencil;
      if (min_requested < 0) {
        work_item_start_reduction =
            std::max(-min_requested, work_item_start_reduction);
      }
      // Check if near end
      i64 max_requested = s.rows(s.rows_size() - 1) + max_stencil;
      if (max_requested > table_num_rows[j]) {
        work_item_end_reduction =
            std::max(max_requested - table_num_rows[j], work_item_end_reduction);
      }
    }
    requested_end_row -= work_item_start_reduction;
    requested_end_row -= work_item_end_reduction;

    requested_start_row -= start_rows_lost;
    requested_end_row -= start_rows_lost;

    start_rows_lost += work_item_start_reduction;
    if ((requested_start_row < requested_end_row) &&
        (requested_end_row - start_rows_lost > 0)) {
      rows.push_back(requested_end_row - start_rows_lost);
    }
  }
  return result;
}
}

MasterImpl::MasterImpl(DatabaseParameters& params)
  : watchdog_awake_(true), db_params_(params), bar_(nullptr) {
  init_glog("scanner_master");
  storage_ =
      storehouse::StorageBackend::make_from_config(db_params_.storage_config);
  set_database_path(params.db_path);
}

MasterImpl::~MasterImpl() {
  trigger_shutdown_.set();
  if (watchdog_thread_.joinable()) {
    watchdog_thread_.join();
  }
  delete storage_;
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

  if (active_job_) {
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
    new_work->mutable_io_item()->set_item_id(-1);
    return grpc::Status::OK;
  }

  // If we do not have any outstanding work, try and create more
  if (unallocated_task_samples_.empty()) {
    // If we have no more samples for this task, try and get another task
    if (next_sample_ == num_samples_) {
      // Check if there are any tasks left
      if (next_task_ < num_tasks_ && task_result_.success()) {
        // More tasks left
        auto sampler = new TaskSampler(
            *table_metas_.get(), job_params_.task_set().tasks(next_task_));
        task_result_ = sampler->validate();
        if (task_result_.success()) {
          next_sample_ = 0;
          num_samples_ = sampler->total_samples();
          next_task_++;
          VLOG(1) << "Tasks left: " << num_tasks_ - next_task_;
          task_samplers_[next_task_ - 1].reset(sampler);
        } else {
          delete sampler;
        }
      }
    }

    // Create more work if possible
    if (next_sample_ < num_samples_) {
      i64 current_task = next_task_ - 1;
      i64 current_sample = next_sample_;

      unallocated_task_samples_.push_front(
          std::make_tuple(current_task, current_sample));
      task_sampler_samples_left_[current_task]++;
      next_sample_++;
    }
  }

  if (unallocated_task_samples_.empty()) {
    // No more work
    new_work->mutable_io_item()->set_item_id(-1);
    return grpc::Status::OK;
  }

  // Grab the next task sample
  std::tuple<i64, i64> task_sample_id = unallocated_task_samples_.back();
  unallocated_task_samples_.pop_back();

  // Get task sampler for our task sample
  assert(next_sample_ <= num_samples_);
  auto& sampler = task_samplers_.at(std::get<0>(task_sample_id));

  // Get the task sample
  task_result_ = sampler->sample_at(std::get<1>(task_sample_id), *new_work);
  if (!task_result_.success()) {
    // Task sampler failed for some reason
    new_work->mutable_io_item()->set_item_id(-1);
    return grpc::Status::OK;
  }
  new_work->mutable_load_work()->set_job_index(std::get<0>(task_sample_id));

  // Track sample assigned to worker
  active_task_samples_[node_info->node_id()].insert(task_sample_id);
  worker_histories_[node_info->node_id()].tasks_assigned += 1;

  return grpc::Status::OK;
}

grpc::Status MasterImpl::FinishedWork(
    grpc::ServerContext* context, const proto::FinishedWorkParameters* params,
    proto::Empty* empty) {
  std::unique_lock<std::mutex> lk(work_mutex_);

  i32 worker_id = params->node_id();
  i64 task_id = params->task_id();
  i64 sample_id = params->sample_id();

  if (!worker_active_[worker_id]) {
    // Technically the task was finished, but we don't count it for now
    // because it would have been reinstered into the work queue
    return grpc::Status::OK;
  }

  auto& worker_samples = active_task_samples_.at(worker_id);

  std::tuple<i64, i64> task_sample = std::make_tuple(task_id, sample_id);
  assert(worker_samples.count(task_sample) > 0);
  worker_samples.erase(task_sample);

  task_sampler_samples_left_[task_id]--;
  worker_histories_[worker_id].tasks_retired += 1;

  i64 active_task = next_task_ - 1;
  // If there are no more samples left in the task, we can get rid of the
  // TaskSampler object (assuming it's not the active task)
  if (task_id != active_task && task_sampler_samples_left_.at(task_id) == 0) {
    task_samplers_.erase(active_task);
  }

  total_samples_used_++;
  if (bar_) {
    bar_->Progressed(total_samples_used_);
  }

  if (total_samples_used_ == total_samples_) {
    assert(next_task_ == num_tasks_);
    {
      std::unique_lock<std::mutex> lock(finished_mutex_);
      finished_ = true;
    }
    finished_cv_.notify_one();
  }

  return grpc::Status::OK;
}

grpc::Status MasterImpl::NewJob(grpc::ServerContext* context,
                                const proto::JobParameters* job_params,
                                proto::Result* job_result) {
  VLOG(1) << "Master received NewJob command";
  job_result->set_success(true);
  set_database_path(db_params_.db_path);

  // Reset job state
  active_job_ = true;
  unallocated_task_samples_.clear();
  next_task_ = 0;
  num_tasks_ = -1;
  task_samplers_.clear();
  task_sampler_samples_left_.clear();
  next_sample_ = 0;
  num_samples_ = -1;
  task_result_.set_success(true);
  active_task_samples_.clear();
  worker_histories_.clear();
  local_ids_.clear();
  local_totals_.clear();
  client_contexts_.clear();
  statuses_.clear();
  replies_.clear();
  rpcs_.clear();
  finished_ = false;

  job_params_.CopyFrom(*job_params);

  const i32 io_item_size = job_params->io_item_size();
  const i32 work_item_size = job_params->work_item_size();
  if (io_item_size > 0 && io_item_size % work_item_size != 0) {
    RESULT_ERROR(job_result,
                 "IO packet size must be a multiple of Work packet size.");
    active_job_ = false;
    return grpc::Status::OK;
  }

  i32 warmup_size = 0;
  i32 total_rows = 0;

  meta_ =
      read_database_metadata(storage_, DatabaseMetadata::descriptor_path());
  DatabaseMetadata meta_copy =
      read_database_metadata(storage_, DatabaseMetadata::descriptor_path());

  validate_task_set(meta_, job_params->task_set(), job_result);
  if (!job_result->success()) {
    // No database changes made at this point, so just return
    active_job_ = false;
    return grpc::Status::OK;
  }

  // Setup table metadata cache
  table_metas_.reset(new TableMetaCache(storage_, meta_));

  // Get output columns from last output op
  std::vector<Column> input_table_columns;
  {
    for (auto& sample : job_params->task_set().tasks(0).samples()) {
      const TableMetadata& table = table_metas_->at(sample.table_name());
      std::vector<Column> table_columns = table.columns();
      for (const std::string& c : sample.column_names()) {
        for (Column& col : table_columns) {
          if (c == col.name()) {
            Column new_col;
            new_col.CopyFrom(col);
            new_col.set_id(input_table_columns.size());
            input_table_columns.push_back(new_col);
          }
        }
      }
    }
  }

  auto& ops = job_params->task_set().ops();
  OpRegistry* op_registry = get_op_registry();
  auto& last_op = ops.Get(ops.size() - 1);
  assert(last_op.name() == "OutputTable");
  std::vector<Column> output_columns;
  for (const auto& eval_input : last_op.inputs()) {
    auto& input_op = ops.Get(eval_input.op_index());
    std::vector<Column> input_columns;
    if (input_op.name() == "InputTable") {
      input_columns = input_table_columns;
    } else {
      OpInfo* input_op_info = op_registry->get_op_info(input_op.name());
      input_columns = input_op_info->output_columns();
    }
    for (const std::string& name : eval_input.columns()) {
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
  proto::JobDescriptor job_descriptor;
  job_descriptor.set_io_item_size(io_item_size);
  job_descriptor.set_work_item_size(work_item_size);
  job_descriptor.set_num_nodes(workers_.size());

  for (size_t i = 0; i < output_columns.size(); ++i) {
    Column* col = job_descriptor.add_columns();
    col->CopyFrom(output_columns[i]);
  }

  auto& tasks = job_params->task_set().tasks();
  job_descriptor.mutable_tasks()->CopyFrom(tasks);

  // Add job name into database metadata so we can look up what jobs have
  // been run
  i32 job_id = meta_.add_job(job_params->job_name());
  job_descriptor.set_id(job_id);
  job_descriptor.set_name(job_params->job_name());

  i64 min_stencil, max_stencil;
  std::tie(min_stencil, max_stencil) =
      determine_stencil_bounds(job_params->task_set());
  total_samples_used_ = 0;
  total_samples_ = 0;
  for (auto& task : job_params->task_set().tasks()) {
    i32 table_id = meta_.add_table(task.output_table_name());
    proto::TableDescriptor table_desc;
    table_desc.set_id(table_id);
    table_desc.set_name(task.output_table_name());
    table_desc.set_timestamp(std::chrono::duration_cast<std::chrono::seconds>(
                                 now().time_since_epoch())
                                 .count());
    // Set columns equal to the last op's output columns
    for (size_t i = 0; i < output_columns.size(); ++i) {
      Column* col = table_desc.add_columns();
      col->CopyFrom(output_columns[i]);
    }
    table_metas_->update(TableMetadata(table_desc));
    std::vector<i64> end_rows;
    Result result = get_task_end_rows(*table_metas_.get(), task, min_stencil,
                                      max_stencil, end_rows);
    if (!result.success()) {
      *job_result = result;
      break;
    }
    total_samples_ += end_rows.size();
    for (i64 r : end_rows) {
      table_desc.add_end_rows(r);
    }
    table_desc.set_job_id(job_id);

    write_table_metadata(storage_, TableMetadata(table_desc));
    table_metas_->update(TableMetadata(table_desc));
  }
  if (!job_result->success()) {
    // No database changes made at this point, so just return
    active_job_ = false;
    return grpc::Status::OK;
  }

  // Write out database metadata so that workers can read it
  write_job_metadata(storage_, JobMetadata(job_descriptor));

  // Setup initial task sampler
  task_result_.set_success(true);
  next_sample_ = 0;
  num_samples_ = 0;
  next_task_ = 0;
  num_tasks_ = job_params->task_set().tasks_size();

  write_database_metadata(storage_, meta_);

  VLOG(1) << "Total tasks: " << num_tasks_;

  if (job_params->show_progress()) {
    bar_.reset(new ProgressBar(total_samples_, ""));
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

  // Wait for all workers to finish
  VLOG(1) << "Waiting for workers to finish";

  {
    // Wait until all workers are done and work has been completed
    while (!finished_) {
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
        next_task_ = num_tasks_;
      }
    }
  }

  {
    std::unique_lock<std::mutex> lock(finished_mutex_);
    finished_cv_.wait(lock, [this] { return finished_; });
  }

  if (!job_result->success()) {
    // Overwrite database metadata with copy from prior to modification
    write_database_metadata(storage_, meta_copy);
  }
  if (!task_result_.success()) {
    job_result->CopyFrom(task_result_);
  } else {
    assert(next_task_ == num_tasks_);
    if (bar_) {
      bar_->Progressed(total_samples_);
    }
  }

  std::fflush(NULL);
  sync();

  active_job_ = false;
  VLOG(1) << "Master finished NewJob";
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
  for (auto& kv : worker_active_) {
    if (kv.second) {
      auto& w = workers_.at(kv.first);
      grpc::ClientContext ctx;
      proto::Empty empty;
      proto::Empty empty2;
      w->PokeWatchdog(&ctx, empty, &empty2);
    }
  }
  return grpc::Status::OK;
}

void MasterImpl::start_watchdog(grpc::Server* server, i32 timeout_ms) {
  watchdog_thread_ = std::thread([this, server, timeout_ms]() {
    double time_since_check = 0;
    // Wait until shutdown is triggered or watchdog isn't woken up
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

void MasterImpl::start_job_on_worker(i32 worker_id,
                                     const std::string& address) {
  proto::JobParameters w_job_params;
  w_job_params.CopyFrom(job_params_);

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
  VLOG(2) << "Sent NewJob command to worker " << worker_id;
}

void MasterImpl::stop_job_on_worker(i32 worker_id) {
  // Place workers active tasks back into the unallocated task samples
  if (active_task_samples_.count(worker_id) > 0) {
    // Keep track of which tasks the worker was assigned
    std::set<i64> tasks;
    // Place workers active tasks back into the unallocated task samples
    for (const std::tuple<i64, i64>& worker_task_sample :
         active_task_samples_.at(worker_id)) {
      unallocated_task_samples_.push_back(worker_task_sample);
      tasks.insert(std::get<0>(worker_task_sample));
    }
    VLOG(1) << "Reassigning worker " << worker_id << "'s "
            << active_task_samples_.at(worker_id).size() << " task samples.";
    active_task_samples_.erase(worker_id);

    // Create samplers for all tasks that are not active
    for (i64 task_id : tasks) {
      if (task_samplers_.count(task_id) == 0) {
        auto sampler = new TaskSampler(
            *table_metas_.get(), job_params_.task_set().tasks(task_id));
        task_result_ = sampler->validate();
        if (task_result_.success()) {
          task_samplers_[task_id].reset(sampler);
        } else {
          delete sampler;
        }
      }
    }
  }

  worker_histories_[worker_id].end_time = now();

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

  stop_job_on_worker(node_id);

  VLOG(1) << "Removing worker " << node_id << " (" << worker_address << ").";

}

}
}
