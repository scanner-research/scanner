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
  return db;
}
}

MachineParameters default_machine_params() {
  MachineParameters machine_params;
  machine_params.num_cpus = std::thread::hardware_concurrency();
  machine_params.num_load_workers = 8;
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

Result Database::start_master(const MachineParameters& machine_params,
                              const std::string& port,
                              bool watchdog) {
  if (master_state_ != nullptr) {
    LOG(WARNING) << "Master already started";
    Result result;
    result.set_success(true);
    return result;
  }
  master_state_.reset(new ServerState);
  internal::DatabaseParameters params =
      machine_params_to_db_params(machine_params, storage_config_, db_path_);

  auto master_service = scanner::internal::get_master_service(params);
  master_state_->service.reset(master_service);
  master_state_->server = start(master_state_->service, port);

  // Setup watchdog
  master_service->start_watchdog(master_state_->server.get(), watchdog);

  Result result;
  result.set_success(true);
  return result;
}

Result Database::start_worker(const MachineParameters& machine_params,
                              const std::string& port,
                              bool watchdog) {
  internal::DatabaseParameters params =
      machine_params_to_db_params(machine_params, storage_config_, db_path_);
  ServerState* s = new ServerState;
  ServerState& state = *s;
  auto worker_service =
      scanner::internal::get_worker_service(params, master_address_, port);
  state.service.reset(worker_service);
  state.server = start(state.service, port);
  worker_states_.emplace_back(s);

  // Setup watchdog
  worker_service->start_watchdog(state.server.get(), watchdog);

  worker_service->register_with_master();

  Result result;
  result.set_success(true);
  return result;
}

Result Database::ingest_videos(const std::vector<std::string>& table_names,
                               const std::vector<std::string>& paths,
                               std::vector<FailedVideo>& failed_videos) {
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

Result Database::new_table(const std::string& table_name,
                           const std::vector<std::string>& columns,
                           const std::vector<std::vector<std::string>>& rows) {
  internal::DatabaseMetadata meta = internal::read_database_metadata(
      storage_.get(), internal::DatabaseMetadata::descriptor_path());

  i32 table_id = meta.add_table(table_name);
  assert(table_id != -1);
  proto::TableDescriptor table_desc;
  table_desc.set_id(table_id);
  table_desc.set_name(table_name);
  table_desc.set_timestamp(
      std::chrono::duration_cast<std::chrono::seconds>(now().time_since_epoch())
          .count());
  for (size_t i = 0; i < columns.size(); ++i) {
    proto::Column* col = table_desc.add_columns();
    col->set_id(i);
    col->set_name(columns[i]);
    col->set_type(proto::ColumnType::Other);
  }

  table_desc.add_end_rows(rows.size());
  table_desc.set_job_id(-1);
  meta.commit_table(table_id);

  internal::write_table_metadata(storage_.get(),
                                 internal::TableMetadata(table_desc));
  internal::write_database_metadata(storage_.get(), meta);

  assert(rows[0].size() == columns.size());
  for (size_t j = 0; j < columns.size(); ++j) {
    const std::string output_path =
        internal::table_item_output_path(table_id, j, 0);

    const std::string output_metadata_path =
        internal::table_item_metadata_path(table_id, j, 0);

    std::unique_ptr<storehouse::WriteFile> output_file;
    storehouse::make_unique_write_file(storage_.get(), output_path,
                                       output_file);

    std::unique_ptr<storehouse::WriteFile> output_metadata_file;
    storehouse::make_unique_write_file(storage_.get(), output_metadata_path,
                                       output_metadata_file);

    u64 num_rows = rows.size();
    s_write(output_metadata_file.get(), num_rows);
    for (size_t i = 0; i < num_rows; ++i) {
      u64 buffer_size = rows[i][j].size();
      s_write(output_metadata_file.get(), buffer_size);
    }
    for (size_t i = 0; i < num_rows; ++i) {
      i64 buffer_size = rows[i][j].size();
      u8* buffer = (u8*)rows[i][j].data();
      s_write(output_file.get(), buffer, buffer_size);
    }

    BACKOFF_FAIL(output_file->save());
    BACKOFF_FAIL(output_metadata_file->save());
  }

  proto::Result result;
  result.set_success(true);
  return result;
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
  if (master_state_ != nullptr) {
    master_state_->server->Wait();
    master_state_.reset(nullptr);
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
  storehouse::StoreResult result = storage_->get_file_info(db_meta_path, info);
  return (result == storehouse::StoreResult::Success);
}
}
