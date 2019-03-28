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

// FIXME(apoms): This should be a configuration option
const i32 MAX_SINK_THREADS = 16;

SaveWorker::SaveWorker(const SaveWorkerArgs& args)
  : node_id_(args.node_id),
    worker_id_(args.worker_id),
    profiler_(args.profiler),
    sink_args_(args.sink_args),
    column_sink_to_table_ids_(args.column_sink_to_table_ids),
    sink_op_idx_(args.sink_op_idxs),
    thread_pool_(MAX_SINK_THREADS) {
  auto setup_start = now();
  // Setup a distinct storage backend for each IO thread

  // Instantiate the sinks and validate that they were properly constructed
  for (size_t i = 0; i < args.sink_factories.size(); ++i) {
    sinks_.emplace_back();
    auto& sink = sinks_.back();
    sink.reset(
        args.sink_factories[i]->new_instance(args.sink_configs[i]));

    sink->set_profiler(&profiler_);

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
}

void SaveWorker::feed(EvalWorkEntry& input_entry) {
  EvalWorkEntry& work_entry = input_entry;

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

  auto write_sink = [&](i32 i) {
    auto write_sink_start = now();

    auto& sink = sinks_.at(i);
    if (auto column_sink = dynamic_cast<ColumnSink*>(sink.get())) {
      column_sink->provide_column_info(compressed, frame_info);
    }
    // Provide index to sink
    BatchedElements inputs(1);
    auto column = work_entry.columns[i];
    inputs[0] = column;
    for (size_t j = 0; j < column.size(); ++j) {
      inputs[0][j].index = work_entry.row_ids[i][j];
    }
    if (inputs[0].size() > 0) {
      sink->write(inputs);
    }

    profiler_.add_interval("load_worker:write_sink_" + std::to_string(i),
                           write_sink_start, now());
  };

  // HACK(apoms): When a pipeline has a large number of sinks, writing each
  // sink serially can take ages. This is a quick hack to overlap writes
  // for pipelines with a large number of sinks. The real fix is to process
  // multiple sinks as different Ops in a DAG scheduler.
  auto sink_start = now();
  std::vector<std::future<void>> futures;
  for (size_t i = 0; i < sinks_.size(); ++i) {
    futures.push_back(thread_pool_.enqueue(write_sink, i));
  }

  for (auto& future : futures) {
    future.wait();
  }

  profiler_.add_interval("save_worker::write_sinks", sink_start, now());

  {
    ProfileBlock _block(&profiler_, "save_worker::cleanup");
    for (size_t out_idx = 0; out_idx < work_entry.columns.size(); ++out_idx) {
      u64 num_elements = static_cast<u64>(work_entry.columns[out_idx].size());
      for (size_t i = 0; i < num_elements; ++i) {
        delete_element(CPU_DEVICE, work_entry.columns[out_idx][i]);
      }
    }
  }
}

void SaveWorker::new_task(i32 job_id, i32 task_id, std::vector<ColumnType> column_types) {
  auto new_task_start = now();

  for (size_t i = 0; i < sinks_.size(); ++i) {
    i32 sink_op_idx = sink_op_idx_[i];
    auto& sink = sinks_[i];
    if (auto column_sink = dynamic_cast<ColumnSink*>(sink.get())) {
      i64 output_table_id = column_sink_to_table_ids_.at(job_id).at(sink_op_idx);
      column_sink->new_task(output_table_id, task_id, column_types);
    }
    sink->new_stream(sink_args_.at(job_id).at(sink_op_idx));
  }

  profiler_.add_interval("save_worker::new_task", new_task_start, now());
}

void SaveWorker::finished() {
  for (auto& sink : sinks_) {
    sink->finished();
  }
}

}
}
