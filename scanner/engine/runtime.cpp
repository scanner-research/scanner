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
    if (is_frame) {
      for (i32 b = 0; b < (i32)column.size(); ++b) {
        Frame* frame = column[b].as_frame();
        delete_buffer(current_handle, frame->data);
        frame->data = dest_buffers[b];
      }
    } else {
      for (i32 b = 0; b < (i32)column.size(); ++b) {
        delete_buffer(current_handle, column[b].buffer);
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

ElementList duplicate_elements(Profiler& profiler, DeviceHandle current_handle,
                               DeviceHandle target_handle, ElementList& column) {
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

  ElementList output_list;
  if (is_frame) {
    for (i32 b = 0; b < (i32)column.size(); ++b) {
      Frame* frame =
          new Frame(column[b].as_frame()->as_frame_info(), dest_buffers[b]);
      insert_frame(output_list, frame);
    }
  } else {
    for (i32 b = 0; b < (i32)column.size(); ++b) {
      insert_element(output_list, dest_buffers[b], sizes[b]);
    }
  }
  return output_list;
}

std::tuple<i64, i64> determine_stencil_bounds(const proto::TaskSet& task_set) {
  i64 min = std::numeric_limits<i64>::max();
  i64 max = std::numeric_limits<i64>::min();

  OpRegistry* op_registry = get_op_registry();
  // Skip input and output table ops
  for (size_t i = 1; i < task_set.ops_size() - 1; ++i) {
    auto& op = task_set.ops(i);
    const auto& op_info = op_registry->get_op_info(op.name());

    std::vector<i32> stencil;
    if (op.stencil_size() > 0) {
      stencil = std::vector<i32>(op.stencil().begin(), op.stencil().end());
    } else {
      stencil = op_info->preferred_stencil();
    }

    min = std::min((i64)stencil[0], min);
    max = std::max((i64)stencil[stencil.size() - 1], max);
  }

  return std::make_tuple(min, max);
}

}
}
