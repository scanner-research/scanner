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

#include "scanner/api/commands.h"
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

#include <thread>

namespace scanner {

namespace {
template <typename T>
std::unique_ptr<grpc::Server> start(T &service, const std::string &port,
                                    bool block) {
  std::string server_address("0.0.0.0:" + port);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(service.get());
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  LOG_IF(FATAL, server.get() == nullptr) << "Failed to start server";
  if (block) {
    server->Wait();
  }
  return std::move(server);
}

bool database_exists(storehouse::StorageBackend *storage,
                     const std::string &database_path) {
  internal::set_database_path(database_path);
  std::string db_meta_path = internal::DatabaseMetadata::descriptor_path();
  storehouse::FileInfo info;
  storehouse::StoreResult result = storage->get_file_info(db_meta_path, info);
  return (result != storehouse::StoreResult::FileDoesNotExist);
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
      for (i64 r : ts.rows) {
        sample->add_rows(r);
      }
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
}

void create_database(storehouse::StorageConfig *storage_config,
                     const std::string &db_path) {
  std::unique_ptr<storehouse::StorageBackend> storage{
      storehouse::StorageBackend::make_from_config(storage_config)};

  if (database_exists(storage.get(), db_path)) {
    LOG(WARNING) << "Can not create database. Database already exists!";
    return;
  }

  internal::set_database_path(db_path);

  internal::DatabaseMetadata meta{};
  internal::write_database_metadata(storage.get(), meta);
}

void destroy_database(storehouse::StorageConfig *storage_config,
                      const std::string &db_path) {
  LOG(FATAL) << "Not implemented yet!";
}

// void ingest_videos(storehouse::StorageConfig *storage_config,
//                    const std::string& db_path,
//                    const std::vector<std::string>& table_names,
//                    const std::vector<std::string>& path) {
// }

// void ingest_images(storehouse::StorageConfig *storage_config,
//                    const std::string& db_path,
//                    const std::string& table_names,
//                    const std::vector<std::string>& paths) {
// }

proto::WorkerParameters default_worker_params() {
  proto::WorkerParameters worker_params;
  worker_params.set_num_cpus(std::thread::hardware_concurrency());
  worker_params.set_num_load_workers(2);
  worker_params.set_num_save_workers(2);
#ifdef HAVE_CUDA
  i32 gpu_count;
  CU_CHECK(cudaGetDeviceCount(&gpu_count));
  for (i32 i = 0; i < gpu_count; ++i) {
    worker_params.add_gpu_ids(i);
  }
#endif
  return worker_params;
}

ServerState start_master(DatabaseParameters &params, bool block) {
  ServerState state;
  state.service.reset(scanner::internal::get_master_service(params));
  state.server = start(state.service, "5001", block);
  return state;
}

ServerState start_worker(DatabaseParameters &db_params,
                         proto::WorkerParameters &worker_params,
                         const std::string &master_address, bool block) {
  ServerState state;
  state.service.reset(scanner::internal::get_worker_service(
      db_params, worker_params, master_address));
  state.server = start(state.service, "5002", block);
  return state;
}

void get_database_info(const std::string &master_address) {}

void get_table_info(const std::string &master_address,
                    const std::string &table_name) {}

void new_job(JobParameters &params) {
  auto channel = grpc::CreateChannel(params.master_address,
                                     grpc::InsecureChannelCredentials());
  std::unique_ptr<proto::Master::Stub> master_ =
      proto::Master::NewStub(channel);

  grpc::ClientContext context;
  proto::JobParameters job_params;
  job_params.set_job_name(params.job_name);
  job_params.set_kernel_instances_per_node(params.kernel_instances_per_node);
  job_params.set_io_item_size(params.io_item_size);
  job_params.set_work_item_size(params.work_item_size);
  proto::TaskSet set = consume_task_set(params.task_set);
  job_params.mutable_task_set()->Swap(&set);
  proto::Empty empty;
  grpc::Status status = master_->NewJob(&context, job_params, &empty);
  LOG_IF(FATAL, !status.ok()) << "Could not contact master server: "
                              << status.error_message();
}
}
