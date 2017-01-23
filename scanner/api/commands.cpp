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
#include "scanner/engine/runtime.h"
#include "scanner/engine/db.h"
#include "scanner/metadata.pb.h"
#include "scanner/engine/rpc.grpc.pb.h"
#include "scanner/engine/rpc.pb.h"

#include <grpc++/server.h>
#include <grpc++/server_builder.h>
#include <grpc++/security/server_credentials.h>
#include <grpc++/security/credentials.h>
#include <grpc++/create_channel.h>

namespace scanner {

namespace {
template <typename T>
std::unique_ptr<grpc::Server> start(T& service, const std::string& port,
                                    bool block) {
  std::string server_address("0.0.0.0:" + port);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(service.get());
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
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


ServerState start_master(DatabaseParameters& params, bool block) {
  ServerState state;
  state.service.reset(scanner::internal::get_master_service(params));
  state.server = start(state.service, "5001", block);
  return state;
}

ServerState start_worker(DatabaseParameters &params,
                  const std::string &master_address, bool block) {
  ServerState state;
  state.service.reset(
      scanner::internal::get_worker_service(params, master_address));
  state.server = start(state.service, "5002", block);
  return state;
}

void new_job(JobParameters& params) {
  auto channel = grpc::CreateChannel(params.master_address,
                                     grpc::InsecureChannelCredentials());
  std::unique_ptr<proto::Master::Stub> master_ =
      proto::Master::NewStub(channel);

  grpc::ClientContext context;
  proto::JobParameters job_params;
  job_params.set_job_name(params.task_set.job_name);
  proto::TaskSet set = consume_task_set(params.task_set);
  job_params.mutable_task_set()->Swap(&set);
  proto::Empty empty;
  printf("before new job\n");
  grpc::Status status = master_->NewJob(&context, job_params, &empty);
  LOG_IF(FATAL, !status.ok()) << "Could not contact master server: "
                              << status.error_message();
}

}
