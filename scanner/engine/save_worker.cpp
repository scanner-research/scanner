/* Copyright 2018 Carnegie Mellon University
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

#include "scanner/engine/metadata.h"
#include "scanner/engine/column_sink.h"
#include "scanner/util/common.h"
#include "scanner/util/storehouse.h"
#include "scanner/video/h264_byte_stream_index_creator.h"

#include "storehouse/storage_backend.h"

#include <glog/logging.h>

using storehouse::StoreResult;
using storehouse::WriteFile;
using storehouse::RandomReadFile;

namespace scanner {
namespace internal {

SaveWorker::SaveWorker(const SaveWorkerArgs& args)
    : node_id_(args.node_id), worker_id_(args.worker_id),
      profiler_(args.profiler), sink_args_(args.sink_args) {
  auto setup_start = now();
  // Setup a distinct storage backend for each IO thread

  // Instantiate the sinks and validate that they were properly constructed
  for (size_t i = 0; i < args.sink_factories.size(); ++i) {
    sinks_.emplace_back();
    auto& sink = sinks_.back();
    sink.reset(
        args.sink_factories[i]->new_instance(args.sink_configs[i]));

    sink->set_profiler(&profiler_);
    sink_op_idx_.push_back(0);

    sink->validate(&args.result);
    VLOG(1) << "Sink finished validation " << args.result.success();
    if (!args.result.success()) {
      LOG(ERROR) << "Sink validate failed: " << args.result.msg();
      THREAD_RETURN_SUCCESS();
    }
  }
  sink_configs_ = args.sink_configs;

  args.profiler.add_interval("setup", setup_start, now());
}

SaveWorker::~SaveWorker() {
  for (auto& sink : sinks_) {
    sink->finished();
  }
}

void SaveWorker::feed(EvalWorkEntry& input_entry) {
  EvalWorkEntry& work_entry = input_entry;

  assert(work_entry.columns.size() == sinks_.size());

  // Write out each output column to an individual data file
  std::vector<bool> compressed;
  std::vector<FrameInfo> frame_info;
  int video_col_idx = 0;
  for (size_t out_idx = 0; out_idx < work_entry.columns.size(); ++out_idx) {
    u64 num_elements = static_cast<u64>(work_entry.columns[out_idx].size());

    if (work_entry.columns[out_idx].size() != num_elements) {
      LOG(FATAL) << "Output layer's element vector has wrong length";
    }

    // Ensure the data is on the CPU
    move_if_different_address_space(profiler_,
                                    work_entry.column_handles[out_idx],
                                    CPU_DEVICE, work_entry.columns[out_idx]);

    compressed.push_back(work_entry.compressed[out_idx]);
    // If this is a video...
    if (work_entry.column_types[out_idx] == ColumnType::Video) {
      // Read frame info column
      assert(work_entry.columns[out_idx].size() > 0);
      frame_info.push_back(work_entry.frame_sizes[video_col_idx]);
      video_col_idx++;
    } else {
      frame_info.emplace_back();
    }
  }

  auto io_start = now();

  auto& sink = sinks_.at(0);
  if (auto column_sink = dynamic_cast<ColumnSink*>(sink.get())) {
    column_sink->provide_column_info(compressed, frame_info);
  }
  sink->write(work_entry.columns);

  profiler_.add_interval("io", io_start, now());

  for (size_t out_idx = 0; out_idx < work_entry.columns.size(); ++out_idx) {
    u64 num_elements = static_cast<u64>(work_entry.columns[out_idx].size());
    for (size_t i = 0; i < num_elements; ++i) {
      delete_element(CPU_DEVICE, work_entry.columns[out_idx][i]);
    }
  }
}

void SaveWorker::new_task(i32 job_id, i32 task_id, i32 output_table_id,
                          std::vector<ColumnType> column_types) {
  auto io_start = now();

  for (auto& sink : sinks_) {
    if (auto column_sink = dynamic_cast<ColumnSink*>(sink.get())) {
      column_sink->new_task(output_table_id, task_id, column_types);
    }
    sink->finished();
    sink->new_stream(sink_args_.at(job_id).at(0));
  }

  profiler_.add_interval("io", io_start, now());
}

}
}
