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
#include "scanner/engine/sampler.h"
#include "scanner/util/common.h"
#include "scanner/util/queue.h"
#include "scanner/video/decoder_automata.h"
#include "scanner/video/video_encoder.h"

#include "hwang/decoder_automata.h"

namespace scanner {
namespace internal {

void move_if_different_address_space(Profiler& profiler,
                                     DeviceHandle current_handle,
                                     DeviceHandle target_handle,
                                     BatchedElements& columns);

///////////////////////////////////////////////////////////////////////////////
/// Worker thread arguments
struct PreEvaluateWorkerArgs {
  // Uniform arguments
  i32 node_id;
  i32 num_cpus;
  i32 decoder_cpus;
  i32 work_packet_size;

  // Per worker arguments
  i32 worker_id;
  DeviceHandle device_handle;
  Profiler& profiler;
};

class PreEvaluateWorker {
 public:
  PreEvaluateWorker(const PreEvaluateWorkerArgs& args);

  void feed(EvalWorkEntry& entry, bool is_first_in_task);

  bool yield(i32 item_size, EvalWorkEntry& output);

 private:
  const i32 node_id_;
  const i32 worker_id_;
  const DeviceHandle device_handle_;
  const i32 num_cpus_;
  const i32 decoder_cpus_;

  Profiler& profiler_;

  i32 last_job_idx_ = -1;

  DeviceHandle decoder_output_handle_;
  std::vector<std::unique_ptr<DecoderAutomata>> decoders_;
  std::vector<std::unique_ptr<hwang::DecoderAutomata>> inplace_decoders_;

  // Continuation state
  bool first_item_;
  bool needs_configure_;
  bool needs_reset_;
  EvalWorkEntry entry_;
  i64 current_row_;
  i64 total_rows_;

  std::vector<std::vector<proto::DecodeArgs>> decode_args_;
};

struct OpArgGroup {
  std::vector<std::string> op_names;
  /// For sampling ops
  // Op -> Job -> slice
  std::map<i64, std::vector<std::vector<proto::SamplingArgs>>> sampling_args;
  /// For slice ops
  // Op -> Job -> slice
  std::map<i64, std::vector<std::vector<i64>>> slice_output_rows;
  /// For unslice ops
  // Op -> Job -> slice
  std::map<i64, std::vector<std::vector<i64>>> unslice_input_rows;
  /// For regular kernels
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
};

struct EvaluateWorkerArgs {
  // Uniform arguments
  i32 node_id;
  std::mutex& startup_lock;
  std::condition_variable& startup_cv;
  i32& startup_count;

  // Per worker arguments
  i32 ki;
  i32 kg;
  OpArgGroup arg_group;

  Profiler& profiler;
  proto::Result& result;
};


class EvaluateWorker {
 public:
  EvaluateWorker(const EvaluateWorkerArgs& args);
  ~EvaluateWorker();

  void new_task(i64 job_idx, i64 task_idx,
                const std::vector<TaskStream>& task_streams);

  void feed(EvalWorkEntry& entry);

  bool yield(i32 item_size, EvalWorkEntry& output);

 private:
  void clear_stencil_cache();

  const i32 node_id_;
  const i32 worker_id_;

  Profiler& profiler_;

  OpArgGroup arg_group_;
  std::vector<DeviceHandle> kernel_devices_;
  std::vector<std::vector<DeviceHandle>> kernel_input_devices_;
  std::vector<std::vector<DeviceHandle>> kernel_output_devices_;
  std::vector<i32> kernel_num_outputs_;
  std::vector<std::unique_ptr<BaseKernel>> kernels_;

  // Used for computing complement of column mapping
  std::vector<std::set<i32>> column_mapping_set_;

  /// Task state
  i64 job_idx_;
  i64 task_idx_;
  i64 slice_group_;
  std::map<i64, std::unique_ptr<DomainSampler>> domain_samplers_;

  // Inputs
  std::vector<std::set<i64>> valid_input_rows_set_;
  std::vector<std::vector<i64>> valid_input_rows_;
  // Tracks which input we should expect next for which column
  std::vector<std::vector<i64>> current_valid_input_idx_;

  // List of row ids of the uutputs to compute
  std::vector<std::set<i64>> compute_rows_set_;
  std::vector<std::vector<i64>> compute_rows_;
  // Tracks which index in compute_rows_ we should expect next
  std::vector<i64> current_compute_idx_;

  // Outputs to keep
  std::vector<std::set<i64>> valid_output_rows_set_;
  std::vector<std::vector<i64>> valid_output_rows_;
  // Tracks which output we should expect next
  std::vector<i64> current_valid_output_idx_;

  // Per kernel -> per input column -> deque of element)
  std::vector<i64> current_element_cache_input_idx_;
  std::vector<std::vector<std::deque<Element>>> element_cache_;
  // Per kernel -> per input column -> device handle
  std::vector<std::vector<DeviceHandle>> element_cache_devices_;
  // Per kernel -> per input column -> deque of row ids
  std::vector<std::vector<std::deque<i64>>> element_cache_row_ids_;

  // Continutation state
  EvalWorkEntry entry_;
  i32 current_input_;
  i32 total_inputs_;

  std::vector<DeviceHandle> final_output_handles_;
  std::vector<std::deque<Element>> final_output_columns_;
  std::vector<std::vector<i64>> final_row_ids_;
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

  void feed(EvalWorkEntry& entry);

  bool yield(EvalWorkEntry& output);

 private:
  Profiler& profiler_;
  std::vector<i32> column_mapping_;
  std::vector<Column> columns_;
  std::set<i32> column_set_;

  DeviceHandle encoder_handle_;
  VideoEncoderType encoder_type_;
  std::vector<std::unique_ptr<VideoEncoder>> encoders_;
  std::vector<bool> frame_size_initialized_;
  std::vector<bool> encoder_configured_;
  std::vector<EncodeOptions> encode_options_;
  std::vector<bool> compression_enabled_;

  // Generator state
  EvalWorkEntry buffered_entry_;
  i64 current_offset_;
  std::deque<EvalWorkEntry> buffered_entries_;
};
}
}
