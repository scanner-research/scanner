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

#include "scanner/api/commands.h"
#include "scanner/api/kernel.h"
#include "scanner/api/evaluator.h"
#include "scanner/video/video_decoder.h"
#include "scanner/util/memory.h"
#include "scanner/engine/rpc.grpc.pb.h"

#include "storehouse/storage_backend.h"

#include <grpc++/server.h>
#include <grpc++/server_builder.h>

#include <string>

namespace scanner {
namespace internal {

///////////////////////////////////////////////////////////////////////////////
/// Work structs - structs used to exchange data between workers during
///   execution of the run command.
struct IOItem {
  // @brief the output table id
  i32 table_id;
  // @brief the unique id for this item in the table
  i64 item_id;
  // @brief the first row in this item
  i64 start_row;
  // @brief the row after the last row in this item
  i64 end_row;
};

struct WorkItem {
  // @brief the index in the IOItem list of the parent io item
  i64 io_item_index;
};

struct EvalWorkEntry {
  i32 io_item_index;
  BatchedColumns columns;
  DeviceHandle buffer_handle;
  // Below only for pre/evaluate/post workers
  bool needs_configure;
  bool needs_reset;
  bool last_in_io_item;
};

proto::Master::Service *get_master_service(DatabaseParameters &param);

proto::Worker::Service *get_worker_service(DatabaseParameters &param,
                                           const std::string &master_address);
}
}
