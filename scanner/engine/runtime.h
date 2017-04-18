/* Copyright 2016 Carnegie Mellon University, NVIDIA Corporation
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

#include "scanner/api/database.h"
#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/engine/kernel_registry.h"
#include "scanner/engine/metadata.h"
#include "scanner/engine/op_registry.h"
#include "scanner/engine/rpc.grpc.pb.h"

#include "storehouse/storage_backend.h"

#include <grpc++/create_channel.h>
#include <grpc++/security/credentials.h>
#include <grpc++/server.h>
#include <grpc++/server_builder.h>

#include <dlfcn.h>
#include <string>
#include <thread>

namespace scanner {
namespace internal {

///////////////////////////////////////////////////////////////////////////////
/// Work structs - structs used to exchange data between workers during
///   execution of the run command.
struct EvalWorkEntry {
  i32 io_item_index;
  BatchedColumns columns;
  std::vector<DeviceHandle> column_handles;
  // Below only for pre/evaluate/post workers
  std::vector<ColumnType> column_types;
  bool needs_configure;
  bool needs_reset;
  bool last_in_io_item;
  i64 warmup_rows;
};

struct DatabaseParameters {
  storehouse::StorageConfig* storage_config;
  std::string db_path;
  i32 num_cpus;
  i32 num_load_workers;
  i32 num_save_workers;
  std::vector<i32> gpu_ids;
};

class MasterImpl;
class WorkerImpl;

MasterImpl* get_master_service(DatabaseParameters& param);

WorkerImpl* get_worker_service(DatabaseParameters& params,
                               const std::string& master_address,
                               const std::string& worker_port);

// Utilities
void move_if_different_address_space(Profiler& profiler,
                                     DeviceHandle current_handle,
                                     DeviceHandle target_handle,
                                     ElementList& column);

void move_if_different_address_space(Profiler& profiler,
                                     DeviceHandle current_handle,
                                     DeviceHandle target_handle,
                                     BatchedColumns& columns);
}
}
