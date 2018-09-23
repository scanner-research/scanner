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
#include "scanner/engine/ingest.h"
#include "scanner/engine/master.h"
#include "scanner/engine/metadata.h"
#include "scanner/engine/rpc.grpc.pb.h"
#include "scanner/engine/rpc.pb.h"
#include "scanner/engine/runtime.h"
#include "scanner/engine/worker.h"
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
std::unique_ptr<grpc::Server> start(T& service, const std::string& port) {
  std::string server_address("0.0.0.0:" + port);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(service.get());
  builder.SetMaxSendMessageSize(1024*1024*1024);
  builder.SetMaxReceiveMessageSize(1024*1024*1024);
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  LOG_IF(FATAL, server.get() == nullptr) << "Failed to start server";
  return std::move(server);
}

internal::DatabaseParameters machine_params_to_db_params(
    const MachineParameters& params, storehouse::StorageConfig* sc,
    const std::string db_path) {
  internal::DatabaseParameters db;
  db.storage_config = sc;
  db.db_path = db_path;
  db.num_cpus = params.num_cpus;
  db.num_load_workers = params.num_load_workers;
  db.num_save_workers = params.num_save_workers;
  db.gpu_ids = params.gpu_ids;
  db.no_workers_timeout = 30;
  db.new_job_retries_limit = 5;
  return db;
}
}

MachineParameters default_machine_params() {
  MachineParameters machine_params;
  machine_params.num_cpus = std::thread::hardware_concurrency();
  machine_params.num_load_workers = 8;
  machine_params.num_save_workers = 4;
#ifdef HAVE_CUDA
  i32 gpu_count;
  CU_CHECK(cudaGetDeviceCount(&gpu_count));
  for (i32 i = 0; i < gpu_count; ++i) {
    machine_params.gpu_ids.push_back(i);
  }
#endif
  return machine_params;
}

Database::Database(storehouse::StorageConfig* storage_config,
                   const std::string& db_path,
                   const std::string& master_address)
  : storage_config_(storage_config),
    storage_(storehouse::StorageBackend::make_from_config(storage_config)),
    db_path_(db_path),
    master_address_(master_address) {
  internal::set_database_path(db_path);
  if (!database_exists()) {
    internal::DatabaseMetadata meta{};
    internal::write_database_metadata(storage_.get(), meta);
    VLOG(1) << "Creating database at " << db_path << "...";
  }
  gpr_set_log_verbosity(GPR_LOG_SEVERITY_ERROR);
}

Database::~Database() {
  wait_for_server_shutdown();
}

Result Database::start_master(const MachineParameters& machine_params,
                              const std::string& port,
                              const std::string& python_dir,
                              bool watchdog,
                              i64 no_workers_timeout,
                              i32 new_job_retries_limit) {
  if (master_server_ != nullptr) {
    LOG(WARNING) << "Master already started";
    Result result;
    result.set_success(true);
    return result;
  }
  internal::DatabaseParameters params =
      machine_params_to_db_params(machine_params, storage_config_, db_path_);
  params.no_workers_timeout = no_workers_timeout;
  params.python_dir = python_dir;
  params.new_job_retries_limit = new_job_retries_limit;

  master_server_.reset(scanner::internal::get_master_service(params, port));
  master_server_->run();

  // Start handling rpcs
  master_thread_ = std::thread([this, watchdog]() {
    master_server_->handle_rpcs(watchdog ? 50000 : -1);
  });

  Result result;
  result.set_success(true);
  return result;
}

Result Database::start_worker(const MachineParameters& machine_params,
                              const std::string& port,
                              const std::string& python_dir,
                              bool watchdog) {
  internal::DatabaseParameters params =
      machine_params_to_db_params(machine_params, storage_config_, db_path_);
  params.python_dir = python_dir;

  ServerState* s = new ServerState;
  ServerState& state = *s;
  auto worker_service =
      scanner::internal::get_worker_service(params, master_address_, port);
  state.service.reset(worker_service);
  state.server = start(state.service, port);
  worker_states_.emplace_back(s);

  // Register shutdown signal handler

  Result register_result = worker_service->register_with_master();
  if (!register_result.success()) {
    return register_result;
  }

  // Setup watchdog
  worker_service->start_watchdog(state.server.get(), watchdog);

  Result result;
  result.set_success(true);
  return result;
}

Result Database::ingest_videos(const std::vector<std::string>& table_names,
                               const std::vector<std::string>& paths,
                               bool inplace,
                               std::vector<FailedVideo>& failed_videos) {
  internal::ingest_videos(storage_config_, db_path_, table_names, paths,
                          inplace,
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
  grpc::Status status = master_->IngestVideos(&context, params, &job_result);
  LOG_IF(FATAL, !status.ok())
      << "Could not contact master server: " << status.error_message();
  for (i32 i = 0; i < job_result.failed_paths().size(); ++i) {
    FailedVideo failed;
    failed.path = job_result.failed_paths(i);
    failed.message = job_result.failed_messages(i);
    failed_videos.push_back(failed);
  }
  return job_result.result();
}

Result Database::delete_table(const std::string& table_name) {
  Result result;
  internal::DatabaseMetadata meta = internal::read_database_metadata(
      storage_.get(), internal::DatabaseMetadata::descriptor_path());

  i32 id = meta.get_table_id(table_name);
  if (id == -1) {
    RESULT_ERROR(&result, "Table %s does not exist", table_name.c_str());
    return result;
  }

  meta.remove_table(id);
  internal::write_database_metadata(storage_.get(), meta);

  internal::TableMetadata table = internal::read_table_metadata(
      storage_.get(), internal::TableMetadata::descriptor_path(id));
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
  if (master_server_ != nullptr) {
    master_thread_.join();
    master_server_.reset(nullptr);
  }
  for (auto& state : worker_states_) {
    state->server->Wait();
  }
  worker_states_.clear();

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
  storehouse::StoreResult result;
  EXP_BACKOFF(storage_->get_file_info(db_meta_path, info), result);
  return (result == storehouse::StoreResult::Success);
}
}
