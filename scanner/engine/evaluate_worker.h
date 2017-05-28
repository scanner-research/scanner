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

#include "scanner/engine/kernel_factory.h"
#include "scanner/engine/runtime.h"
#include "scanner/util/common.h"
#include "scanner/util/queue.h"
#include "scanner/video/decoder_automata.h"
#include "scanner/video/video_encoder.h"

namespace scanner {
namespace internal {

void move_if_different_address_space(Profiler& profiler,
                                     DeviceHandle current_handle,
                                     DeviceHandle target_handle,
                                     BatchedColumns& columns);

///////////////////////////////////////////////////////////////////////////////
/// Worker thread arguments
struct PreEvaluateWorkerArgs {
  // Uniform arguments
  i32 node_id;
  i32 num_cpus;

  // Per worker arguments
  i32 worker_id;
  DeviceHandle device_handle;
  Profiler& profiler;
};

class PreEvaluateWorker {
 public:
  PreEvaluateWorker(const PreEvaluateWorkerArgs& args);

  void feed(std::tuple<IOItem, EvalWorkEntry>& entry);

  bool yield(i32 item_size, std::tuple<IOItem, EvalWorkEntry>& output);

 private:
  const i32 node_id_;
  const i32 worker_id_;
  const DeviceHandle device_handle_;
  const i32 num_cpus_;

  Profiler& profiler_;

  i32 last_table_id_ = -1;
  i32 last_end_row_ = -1;
  i32 last_item_id_ = -1;

  DeviceHandle decoder_output_handle_;
  std::vector<std::unique_ptr<DecoderAutomata>> decoders_;

  // Continuation state
  bool first_item_;
  bool needs_configure_;
  bool needs_reset_;
  std::tuple<IOItem, EvalWorkEntry> entry_;
  i64 current_row_;
  i64 total_rows_;

  std::vector<std::vector<proto::DecodeArgs>> decode_args_;
};

struct EvaluateWorkerArgs {
  // Uniform arguments
  i32 node_id;

  // Per worker arguments
  i32 ki;
  i32 kg;
  std::vector<std::tuple<KernelFactory*, KernelConfig>> kernel_factories;
  std::vector<std::vector<std::tuple<i32, std::string>>> live_columns;
  // Discarded after kernel use
  std::vector<std::vector<i32>> dead_columns;
  // Discarded immediately after kernel execute
  std::vector<std::vector<i32>> unused_outputs;
  // Index in columns for inputs
  std::vector<std::vector<i32>> column_mapping;
  // Stencil needed by kernels
  std::vector<std::vector<i32>> kernel_stencils;
  // Batch size needed by kernels
  std::vector<i32> kernel_batch_sizes;

  Profiler& profiler;
  proto::Result& result;
};


class EvaluateWorker {
 public:
  EvaluateWorker(const EvaluateWorkerArgs& args);

  void new_task(const std::vector<TaskStream>& task_streams);

  void feed(std::tuple<IOItem, EvalWorkEntry>& entry);

  bool yield(i32 item_size, std::tuple<IOItem, EvalWorkEntry>& output);

 private:
  const i32 node_id_;
  const i32 worker_id_;

  Profiler& profiler_;

  std::vector<std::tuple<KernelFactory*, KernelConfig>> kernel_factories_;
  std::vector<DeviceHandle> kernel_devices_;
  std::vector<i32> kernel_num_outputs_;
  std::vector<std::unique_ptr<BaseKernel>> kernels_;

  std::vector<std::vector<i32>> dead_columns_;
  std::vector<std::vector<i32>> unused_outputs_;
  std::vector<std::vector<i32>> column_mapping_;
  std::vector<std::vector<i32>> kernel_stencils_;
  std::vector<i32> kernel_batch_sizes_;

  // Used for computing complement of column mapping
  std::vector<std::set<i32>> column_mapping_set_;

  // Task state
  std::vector<std::set<i64>> valid_output_rows_set_;
  std::vector<std::vector<i64>> valid_output_rows_;
  std::vector<i64> current_valid_idx_;
  // Per kernel -> per input column -> deque of (row, element)
  std::vector<std::vector<std::deque<std::tuple<i64, Element>>>> stencil_cache_;

  // Continutation state
  std::tuple<IOItem, EvalWorkEntry> entry_;
  i32 current_input_;
  i32 total_inputs_;

  i64 outputs_yielded_;
  std::vector<DeviceHandle> final_output_handles_;
  std::vector<std::deque<Element>> final_output_columns_;
  std::vector<i64> final_row_ids_;
};

struct ColumnCompressionOptions {
  std::string codec;
  std::map<std::string, std::string> options;
};

struct PostEvaluateWorkerArgs {
  // Uniform arguments
  i32 node_id;

  // Per worker arguments
  i32 id;
  Profiler& profiler;
  // Index in columns for inputs
  std::vector<i32> column_mapping;
  std::vector<Column> columns;
  std::vector<ColumnCompressionOptions> column_compression;
};

class PostEvaluateWorker {
 public:
  PostEvaluateWorker(const PostEvaluateWorkerArgs& args);

  void feed(std::tuple<IOItem, EvalWorkEntry>& entry);

  bool yield(std::tuple<IOItem, EvalWorkEntry>& output);

 private:
  Profiler& profiler_;
  std::vector<i32> column_mapping_;
  std::vector<Column> columns_;
  std::set<i32> column_set_;

  DeviceHandle encoder_handle_;
  VideoEncoderType encoder_type_;
  std::vector<std::unique_ptr<VideoEncoder>> encoders_;
  std::vector<bool> encoder_configured_;
  std::vector<EncodeOptions> encode_options_;
  std::vector<bool> compression_enabled_;

  // Generator state
  EvalWorkEntry buffered_entry_;
  i64 current_offset_;
  std::deque<std::tuple<IOItem, EvalWorkEntry>> buffered_entries_;
};
}
}
