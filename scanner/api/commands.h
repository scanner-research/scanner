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

#include "scanner/api/evaluator.h"
#include "scanner/util/memory.h"
#include "storehouse/storage_backend.h"

#include <grpc++/server.h>
#include "scanner/engine/rpc.grpc.pb.h"

#include <string>

namespace scanner {
///////////////////////////////////////////////////////////////////////////////
/// Database management
void create_database(storehouse::StorageConfig *storage_config,
                     const std::string &db_path);

void destroy_database(storehouse::StorageConfig *storage_config,
                      const std::string &db_path);

///////////////////////////////////////////////////////////////////////////////
/// Ingest
void ingest_videos(storehouse::StorageConfig *storage_config,
                   const std::string& db_path,
                   const std::vector<std::string>& table_names,
                   const std::vector<std::string>& path);

void ingest_images(storehouse::StorageConfig *storage_config,
                   const std::string& db_path,
                   const std::string& table_name,
                   const std::vector<std::string>& paths);

///////////////////////////////////////////////////////////////////////////////
/// Node setup
struct DatabaseParameters {
  storehouse::StorageConfig* storage_config;
  MemoryPoolConfig memory_pool_config;
  std::string db_path;
};

struct ServerState {
  std::unique_ptr<grpc::Server> server;
  std::unique_ptr<grpc::Service> service;
};

ServerState start_master(DatabaseParameters &params, bool block = true);

ServerState start_worker(DatabaseParameters &params,
                           const std::string &master_address,
                           bool block = true);

///////////////////////////////////////////////////////////////////////////////
/// Job submission
struct TableSample {
  std::string table_name;
  std::vector<std::string> column_names;
  std::vector<i64> rows;
};

struct Task {
  std::string output_table_name;
  std::vector<TableSample> samples;
};

struct TaskSet {
  std::vector<Task> tasks;
  Evaluator* output_evaluator;
};

struct JobParameters {
  std::string master_address;
  std::string job_name;
  TaskSet task_set;
};

void new_job(JobParameters& params);

///////////////////////////////////////////////////////////////////////////////
/// Metadata access
// class TableInformation {
//  public:
//   TableInformation(i64 rows,
//                    const std::vector<std::string>& sample_job_names,
//                    const std::vector<std::string>& sample_table_names,
//                    const std::vector<std::vector<std::string>>& sample_columns,
//                    const std::vector<std::vector<i64>>& sample_rows);

//   i64 num_rows() const;

//   const std::vector<TableSample>& samples() const;

//  private:
//   i64 rows_;
//   std::vector<TableSample> samples_;
// };

// class DatabaseInformation {
//  public:
//   JobInformation(const std::string& dataset_name, const std::string& job_name,
//                  storehouse::StorageBackend* storage);

//   const std::vector<std::string>& table_names() const;

//   const i32 table_id(const std::string& name) const;

//  private:
//   std::string dataset_name_;
//   std::string job_name_;
//   storehouse::StorageBackend* storage_;
//   std::vector<std::string> table_names_;
//   std::vector<std::string> column_names_;
//   std::map<std::string, TableInformation> tables_;
// };

// DatabaseInformation get_database_info(storehouse::StorageConfig *storage_config,
//                                       const std::string &db_path);

// void get_table_info(storehouse::StorageConfig* storage_config,
//                     const std::string& db_path,
//                     const std::string& table_name);


}
