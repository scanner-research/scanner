/* Copyright 2016 Carnegie Mellon University, Stanford University
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

#include "scanner/api/database.h"
#include "scanner/engine/runtime.h"
#include "scanner/engine/ingest.h"
#include "scanner/engine/db.h"
#include "scanner/engine/rpc.grpc.pb.h"
#include "scanner/engine/rpc.pb.h"
#include "scanner/engine/runtime.h"
#include "scanner/metadata.pb.h"
#include "scanner/util/cuda.h"

#include <grpc++/create_channel.h>
#include <grpc++/security/credentials.h>
#include <grpc++/security/server_credentials.h>
#include <grpc++/server.h>
#include <grpc++/server_builder.h>
#include <grpc/support/log.h>

#include <thread>

namespace scanner {

namespace {
template <typename T>
std::unique_ptr<grpc::Server> start(T &service, const std::string &port) {
  std::string server_address("0.0.0.0:" + port);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(service.get());
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  LOG_IF(FATAL, server.get() == nullptr) << "Failed to start server";
  return std::move(server);
}

proto::TaskSet consume_task_set(TaskSet &ts) {
  proto::TaskSet task_set;
  // Parse tasks
  for (Task &t : ts.tasks) {
    proto::Task *task = task_set.add_tasks();
    task->set_output_table_name(t.output_table_name);
    for (TableSample &ts : t.samples) {
      proto::TableSample *sample = task->add_samples();
      sample->set_table_name(ts.table_name);
      for (std::string &s : ts.column_names) {
        sample->add_column_names(s);
      }
      sample->set_sampling_function(ts.sampling_function);
      sample->set_sampling_args(ts.sampling_args.data(),
                                ts.sampling_args.size());
    }
  }
  // Parse ops
  std::map<Op *, std::vector<Op *>> edges; // parent -> child
  std::map<Op *, i32> in_edges_left;              // parent -> child
  Op *start_node = nullptr;
  {
    // Find all edges
    std::set<Op *> explored_nodes;
    std::vector<Op *> stack;
    stack.push_back(ts.output_op);
    while (!stack.empty()) {
      Op *c = stack.back();
      stack.pop_back();
      explored_nodes.insert(c);

      if (c->get_name() == "InputTable") {
        assert(start_node == nullptr);
        start_node = c;
        continue;
      }
      for (const OpInput &input : c->get_inputs()) {
        Op *parent_eval = input.get_op();
        edges[parent_eval].push_back(c);
        in_edges_left[c] += 1;

        if (explored_nodes.count(parent_eval) > 0)
          continue;
        stack.push_back(parent_eval);
      }
    }
  }
  std::vector<Op *> sorted_ops;
  std::map<Op *, i32> op_index;
  {
    // Perform topological sort
    std::vector<Op *> stack;
    stack.push_back(start_node);
    while (!stack.empty()) {
      Op *curr = stack.back();
      stack.pop_back();

      sorted_ops.push_back(curr);
      op_index.insert({curr, sorted_ops.size() - 1});
      for (Op *child : edges[curr]) {
        i32 &edges_left = in_edges_left[child];
        edges_left -= 1;
        if (edges_left == 0) {
          stack.push_back(child);
        }
      }
    }
  }
  assert(sorted_ops.size() == in_edges_left.size() + 1);
  // Translate sorted ops into serialized task set
  for (Op *eval : sorted_ops) {
    proto::Op *proto_eval = task_set.add_ops();
    proto_eval->set_name(eval->get_name());
    proto_eval->set_device_type(eval->get_device_type());
    proto_eval->set_kernel_args(eval->get_args(), eval->get_args_size());
    for (const OpInput &input : eval->get_inputs()) {
      proto::OpInput *proto_input = proto_eval->add_inputs();
      i32 parent_index;
      if (input.get_op() == nullptr) {
        parent_index = -1;
      } else {
        parent_index = op_index.at(input.get_op());
      }
      proto_input->set_op_index(parent_index);
      for (const std::string &column_name : input.get_columns()) {
        proto_input->add_columns(column_name);
      }
    }
  }

  return task_set;
}

internal::DatabaseParameters
machine_params_to_db_params(const MachineParameters &params,
                            storehouse::StorageConfig *sc,
                            const std::string db_path) {
  internal::DatabaseParameters db;
  db.storage_config = sc;
  db.db_path = db_path;
  db.num_cpus = params.num_cpus;
  db.num_load_workers = params.num_load_workers;
  db.num_save_workers = params.num_save_workers;
  db.gpu_ids = params.gpu_ids;
  return db;
}
}

MachineParameters default_machine_params() {
  MachineParameters machine_params;
  machine_params.num_cpus = std::thread::hardware_concurrency();
  machine_params.num_load_workers = 2;
  machine_params.num_save_workers = 2;
#ifdef HAVE_CUDA
  i32 gpu_count;
  CU_CHECK(cudaGetDeviceCount(&gpu_count));
  for (i32 i = 0; i < gpu_count; ++i) {
    machine_params.gpu_ids.push_back(i);
  }
#endif
  return machine_params;
}

Database::Database(storehouse::StorageConfig *storage_config,
                   const std::string &db_path,
                   const std::string &master_address)
    : storage_config_(storage_config),
      storage_(storehouse::StorageBackend::make_from_config(storage_config)),
      db_path_(db_path), master_address_(master_address) {

  internal::set_database_path(db_path);
  if (!database_exists()) {
    internal::DatabaseMetadata meta{};
    internal::write_database_metadata(storage_.get(), meta);
    VLOG(1) << "Creating database at " << db_path << "...";
  }
  gpr_set_log_verbosity(GPR_LOG_SEVERITY_ERROR);
}

Result Database::start_master(const MachineParameters& machine_params) {
  if (master_state_.service.get() != nullptr) {
    LOG(WARNING) << "Master already started";
    Result result;
    result.set_success(true);
    return result;
  }
  internal::DatabaseParameters params =
      machine_params_to_db_params(machine_params, storage_config_, db_path_);
  master_state_.service.reset(scanner::internal::get_master_service(params));
  master_state_.server = start(master_state_.service, "15555");

  Result result;
  result.set_success(true);
  return result;
}

Result Database::start_worker(const MachineParameters& machine_params) {
  internal::DatabaseParameters params =
      machine_params_to_db_params(machine_params, storage_config_, db_path_);
  worker_states_.emplace_back();
  ServerState &state = worker_states_.back();
  state.service.reset(
      scanner::internal::get_worker_service(params, master_address_));
  state.server = start(state.service, "15556");

  Result result;
  result.set_success(true);
  return result;
}

Result Database::ingest_videos(const std::vector<std::string> &table_names,
                               const std::vector<std::string> &paths,
                               std::vector<FailedVideo> &failed_videos) {
  internal::ingest_videos(storage_config_, db_path_, table_names, paths,
                          failed_videos);
  Result result;
  result.set_success(true);
  return result;

  auto channel =
      grpc::CreateChannel(master_address_, grpc::InsecureChannelCredentials());
  std::unique_ptr<proto::Master::Stub> master_ =
      proto::Master::NewStub(channel);

  grpc::ClientContext context;
  proto::IngestParameters params;
  for (auto& t : table_names) {
    params.add_table_names(t);
  }
  for (auto& p : paths) {
    params.add_video_paths(p);
  }
  proto::IngestResult job_result;
  grpc::Status status =
      master_->IngestVideos(&context, params, &job_result);
  LOG_IF(FATAL, !status.ok()) << "Could not contact master server: "
                              << status.error_message();
  for (i32 i = 0; i < job_result.failed_paths().size(); ++i) {
    FailedVideo failed;
    failed.path = job_result.failed_paths(i);
    failed.message = job_result.failed_messages(i);
    failed_videos.push_back(failed);
  }
  return job_result.result();
}

Result Database::new_job(JobParameters &params) {
  auto channel =
      grpc::CreateChannel(master_address_, grpc::InsecureChannelCredentials());
  std::unique_ptr<proto::Master::Stub> master_ =
      proto::Master::NewStub(channel);

  grpc::ClientContext context;
  proto::JobParameters job_params;
  job_params.set_job_name(params.job_name);
  job_params.set_pipeline_instances_per_node(params.pipeline_instances_per_node);
  job_params.set_work_item_size(params.work_item_size);
  proto::TaskSet set = consume_task_set(params.task_set);
  job_params.mutable_task_set()->Swap(&set);
  Result job_result;
  grpc::Status status = master_->NewJob(&context, job_params, &job_result);
  LOG_IF(FATAL, !status.ok()) << "Could not contact master server: "
                              << status.error_message();

  return job_result;
}

Result Database::shutdown_master() {
  LOG(FATAL) << "Not implemented yet!";

  Result result;
  result.set_success(true);
  return result;
}

Result Database::shutdown_worker() {
  LOG(FATAL) << "Not implemented yet!";

  Result result;
  result.set_success(true);
  return result;
}

Result Database::wait_for_server_shutdown() {
  if (master_state_.server.get() != nullptr) {
    master_state_.server->Wait();
  }
  for (ServerState& state : worker_states_) {
    state.server->Wait();
  }

  Result result;
  result.set_success(true);
  return result;
}

Result Database::destroy_database() {
  LOG(FATAL) << "Not implemented yet!";

  Result result;
  result.set_success(true);
  return result;
}

bool Database::database_exists() {
  internal::set_database_path(db_path_);
  std::string db_meta_path = internal::DatabaseMetadata::descriptor_path();
  storehouse::FileInfo info;
  storehouse::StoreResult result = storage_->get_file_info(db_meta_path, info);
  return (result != storehouse::StoreResult::FileDoesNotExist);
}

}
