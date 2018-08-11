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
#include "scanner/util/queue.h"

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
  i64 table_id;
  i64 job_index;
  i64 task_index;
  std::vector<std::vector<i64>> row_ids;
  BatchedElements columns;
  std::vector<DeviceHandle> column_handles;
  // Below only for pre/evaluate/post workers
  std::vector<bool> inplace_video;
  std::vector<ColumnType> column_types;
  bool needs_configure;
  bool needs_reset;
  bool last_in_io_packet;
  // Only for pre worker
  std::vector<proto::VideoDescriptor::VideoCodecType> video_encoding_type;
  bool first;
  bool last_in_task;
  // For save and pre worker
  std::vector<FrameInfo> frame_sizes;
  std::vector<bool> compressed;
};

// Contains the row indices that a Op will see for the given task
struct TaskStream {
  i64 slice_group;
  // This is the set of input rows that the Op needs to keep.
  std::vector<i64> valid_input_rows;
  // This is the set of input rows it should process and produce
  // outputs for.
  std::vector<i64> compute_input_rows;
  // This is the set of outputs that it should pass along (those
  // not in this set should be immediately discarded). Needed
  // to support bounded state operations which must produce output
  // for elements, but that are not necessary for downstream operations
  std::vector<i64> valid_output_rows;
};

using LoadInputQueue =
    Queue<std::tuple<i32, std::deque<TaskStream>, LoadWorkEntry>>;
using EvalQueue =
    Queue<std::tuple<std::deque<TaskStream>, EvalWorkEntry>>;
using OutputEvalQueue =
    Queue<std::tuple<i32, EvalWorkEntry>>;
using SaveInputQueue =
    Queue<std::tuple<i32, EvalWorkEntry>>;
using SaveOutputQueue =
    Queue<std::tuple<i32, i64, i64>>;

struct DatabaseParameters {
  storehouse::StorageConfig* storage_config;
  std::string db_path;
  i32 num_cpus;
  i32 num_load_workers;
  i32 num_save_workers;
  std::vector<i32> gpu_ids;
  i64 no_workers_timeout; // in seconds
  std::string python_dir;
  i32 new_job_retries_limit;
};

class MasterServerImpl;
class WorkerImpl;

MasterServerImpl* get_master_service(DatabaseParameters& param, const std::string& port);

WorkerImpl* get_worker_service(DatabaseParameters& params,
                               const std::string& master_address,
                               const std::string& worker_port);

// Utilities
void move_if_different_address_space(Profiler& profiler,
                                     DeviceHandle current_handle,
                                     DeviceHandle target_handle,
                                     Elements& column);

void move_if_different_address_space(Profiler& profiler,
                                     DeviceHandle current_handle,
                                     DeviceHandle target_handle,
                                     BatchedElements& columns);

Elements copy_elements(Profiler& profiler, DeviceHandle current_handle,
                          DeviceHandle target_handle, Elements& column);

Elements copy_or_ref_elements(Profiler& profiler,
                                 DeviceHandle current_handle,
                                 DeviceHandle target_handle,
                                 Elements& column);
}
}
