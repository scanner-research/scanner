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

#include "scanner/engine/save_worker.h"

#include "scanner/engine/db.h"
#include "scanner/util/common.h"
#include "scanner/util/storehouse.h"

#include "storehouse/storage_backend.h"

#include <glog/logging.h>

using storehouse::StoreResult;
using storehouse::WriteFile;
using storehouse::RandomReadFile;

namespace scanner {
namespace internal {

void *save_thread(void *arg) {
  SaveThreadArgs &args = *reinterpret_cast<SaveThreadArgs *>(arg);

  auto setup_start = now();

  // Setup a distinct storage backend for each IO thread
  storehouse::StorageBackend *storage =
      storehouse::StorageBackend::make_from_config(args.storage_config);

  args.profiler.add_interval("setup", setup_start, now());

  while (true) {
    auto idle_start = now();

    std::tuple<IOItem, EvalWorkEntry> entry;
    args.input_work.pop(entry);
    IOItem &io_item = std::get<0>(entry);
    EvalWorkEntry &work_entry = std::get<1>(entry);

    if (work_entry.io_item_index == -1) {
      break;
    }

    VLOG(2) << "Save (N/KI: " << args.node_id << "/" << args.id
              << "): processing item " << work_entry.io_item_index;

    args.profiler.add_interval("idle", idle_start, now());

    auto work_start = now();

    // Write out each output column to an individual data file
    for (size_t out_idx = 0; out_idx < work_entry.columns.size(); ++out_idx) {
      u64 num_rows = static_cast<u64>(work_entry.columns[out_idx].rows.size());

      const std::string output_path = table_item_output_path(
          io_item.table_id(), out_idx, io_item.item_id());

      auto io_start = now();

      WriteFile *output_file = nullptr;
      BACKOFF_FAIL(storage->make_write_file(output_path, output_file));

      if (work_entry.columns[out_idx].rows.size() != num_rows) {
        LOG(FATAL) << "Output layer's row vector has wrong length";
      }

      if (!work_entry.column_handles[out_idx].is_same_address_space(
              CPU_DEVICE)) {
        std::vector<u8 *> dest_buffers, src_buffers;
        std::vector<size_t> sizes;
        size_t total_size = 0;
        for (i32 f = 0; f < num_rows; ++f) {
          Row &row = work_entry.columns[out_idx].rows[f];
          total_size += row.size;
        }

        if (num_rows > 0) {
          u8 *output_block = new_block_buffer(CPU_DEVICE, total_size, num_rows);
          for (i32 f = 0; f < num_rows; ++f) {
            Row &row = work_entry.columns[out_idx].rows[f];
            size_t size = row.size;
            u8 *src_buffer = row.buffer;
            u8 *dest_buffer = output_block;

            dest_buffers.push_back(dest_buffer);
            src_buffers.push_back(src_buffer);
            sizes.push_back(size);

            output_block += size;
          }

          memcpy_vec(dest_buffers, CPU_DEVICE, src_buffers,
                     work_entry.column_handles[out_idx], sizes);

          for (i32 f = 0; f < num_rows; ++f) {
            delete_buffer(work_entry.column_handles[out_idx] , src_buffers[f]);
            work_entry.columns[out_idx].rows[f].buffer = dest_buffers[f];
          }
        }
      }

      // Write number of rows in the file
      s_write(output_file, num_rows);
      // Write out all output sizes first so we can easily index into the file
      i64 size_written = 0;
      for (size_t i = 0; i < num_rows; ++i) {
        i64 buffer_size = work_entry.columns[out_idx].rows[i].size;
        s_write(output_file, buffer_size);
        size_written += sizeof(i64);
      }
      // Write actual output data
      for (size_t i = 0; i < num_rows; ++i) {
        i64 buffer_size = work_entry.columns[out_idx].rows[i].size;
        u8 *buffer = work_entry.columns[out_idx].rows[i].buffer;
        s_write(output_file, buffer, buffer_size);
        size_written += buffer_size;
      }

      BACKOFF_FAIL(output_file->save());

      // TODO(apoms): For now, all evaluators are expected to return CPU
      //   buffers as output so just assume CPU
      for (size_t i = 0; i < num_rows; ++i) {
        delete_buffer(CPU_DEVICE, work_entry.columns[out_idx].rows[i].buffer);
      }

      delete output_file;

      args.profiler.add_interval("io", io_start, now());
      args.profiler.increment("io_write", size_written);
    }

    VLOG(2) << "Save (N/KI: " << args.node_id << "/" << args.id
              << "): finished item " << work_entry.io_item_index;

    args.profiler.add_interval("task", work_start, now());

    args.retired_items++;
  }

  VLOG(1) << "Save (N/KI: " << args.node_id << "/" << args.id
            << "): thread finished ";

  // Cleanup
  delete storage;

  THREAD_RETURN_SUCCESS();
}
}
}
