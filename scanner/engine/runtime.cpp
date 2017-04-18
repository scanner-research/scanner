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

#include "scanner/engine/runtime.h"
#include "scanner/engine/master.h"
#include "scanner/engine/worker.h"

namespace scanner {
namespace internal {

MasterImpl* get_master_service(DatabaseParameters& param) {
  return new MasterImpl(param);
}

WorkerImpl* get_worker_service(DatabaseParameters& params,
                               const std::string& master_address,
                               const std::string& worker_port) {
  return new WorkerImpl(params, master_address, worker_port);
}

void move_if_different_address_space(Profiler& profiler,
                                     DeviceHandle current_handle,
                                     DeviceHandle target_handle,
                                     ElementList& column) {
  if (!current_handle.is_same_address_space(target_handle) &&
      column.size() > 0) {
    bool is_frame = column[0].is_frame;

    std::vector<u8*> src_buffers;
    std::vector<u8*> dest_buffers;
    std::vector<size_t> sizes;
    if (is_frame) {
      for (i32 b = 0; b < (i32)column.size(); ++b) {
        Frame* frame = column[b].as_frame();
        src_buffers.push_back(frame->data);
        sizes.push_back(frame->size());
      }
    } else {
      for (i32 b = 0; b < (i32)column.size(); ++b) {
        src_buffers.push_back(column[b].buffer);
        sizes.push_back(column[b].size);
      }
    }

    size_t total_size = 0;
    for (i32 b = 0; b < (i32)column.size(); ++b) {
      total_size += sizes[b];
    }

    u8* block = new_block_buffer(target_handle, total_size, column.size());
    for (i32 b = 0; b < (i32)column.size(); ++b) {
      size_t size = sizes[b];
      dest_buffers.push_back(block);
      block += size;
    }

    auto memcpy_start = now();
    memcpy_vec(dest_buffers, target_handle, src_buffers, current_handle, sizes);
    profiler.add_interval("memcpy", memcpy_start, now());

    auto delete_start = now();
    for (i32 b = 0; b < (i32)column.size(); ++b) {
      column[b].buffer = dest_buffers[b];
    }
    if (is_frame) {
      for (i32 b = 0; b < (i32)column.size(); ++b) {
        Frame* frame = column[b].as_frame();
        frame->data = dest_buffers[b];
      }
    } else {
      for (i32 b = 0; b < (i32)column.size(); ++b) {
        column[b].buffer = dest_buffers[b];
      }
    }
  }
}

void move_if_different_address_space(Profiler& profiler,
                                     DeviceHandle current_handle,
                                     DeviceHandle target_handle,
                                     BatchedColumns& columns) {
  for (i32 i = 0; i < (i32)columns.size(); ++i) {
    ElementList& column = columns[i];
    move_if_different_address_space(profiler, current_handle, target_handle,
                                    column);
  }
}

}
}
