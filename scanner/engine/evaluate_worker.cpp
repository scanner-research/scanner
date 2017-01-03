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

#include "scanner/engine/evaluate_worker.h"

namespace scanner {
void* pre_evaluate_thread(void* arg) {
  PreEvaluateThreadArgs& args = *reinterpret_cast<PreEvaluateThreadArgs*>(arg);

  i32 rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  i64 work_item_size = rows_per_work_item();

  i32 last_table_id = -1;
  i32 last_end_row = -1;
  i32 last_item_id = -1;
  while (true) {
    auto idle_start = now();
    // Wait for next work item to process
    EvalWorkEntry work_entry;
    args.input_work.pop(work_entry);

    if (work_entry.io_item_index == -1) {
      break;
    }

    LOG(INFO) << "Pre-evaluate (N/PU: " << rank << "/" << args.id << "): "
              << "processing item " << work_entry.io_item_index;

    args.profiler.add_interval("idle", idle_start, now());

    auto work_start = now();

    const IOItem& io_item = args.io_items[work_entry.io_item_index];
    const BatchConfig& batch_config = args.metadata.at(io_item.table_id);

    bool needs_configure = !(io_item.table_id == last_table_id);
    bool needs_reset = !(io_item.item_id == last_item_id ||
                         (io_item.table_id == last_table_id &&
                          io_item.start_row == last_end_row));
    last_table_id = io_item.table_id;
    last_end_row = io_item.end_row;
    last_item_id = io_item.item_id;

    // Split up a work entry into work item size chunks
    i64 total_rows = work_entry.columns[0].rows.size();
    std::vector<EvalWorkEntry> work_items;
    for (i64 r = 0; r < total_rows; r += work_item_size) {
      work_items.emplace_back();
      EvalWorkEntry& entry = work_items.back();
      entry.io_item_index = work_entry.io_item_index;
      entry.column_names = work_entry.column_names;
      entry.buffer_handle = work_entry.buffer_handle;
      entry.needs_configure = false;
      entry.needs_reset = false;
      entry.last_in_io_item = false;

      entry.columns.resize(work_entry.columns.size());
      for (size_t c = 0; c < work_entry.columns.size(); ++c) {
        entry.columns[c].rows =
            std::vector<Row>(work_entry.columns[c].rows.begin() + r,
                             work_entry.columns[c].rows.begin() +
                                 std::min(r + work_item_size, total_rows));
      }
    }
    assert(!work_items.empty());
    work_items.front().needs_configure = needs_configure;
    work_items.front().needs_reset = needs_reset;
    work_items.back().last_in_io_item = true;

    for (EvalWorkEntry& output_work_entry : work_items) {
      args.output_work.push(output_work_entry);
    }
  }

  THREAD_RETURN_SUCCESS();
}

void* evaluate_thread(void* arg) {
  EvaluateThreadArgs& args = *reinterpret_cast<EvaluateThreadArgs*>(arg);

  auto setup_start = now();

  i32 rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  assert(args.evaluator_factories.size() == args.evaluator_configs.size());
  std::vector<EvaluatorCapabilities> evaluator_caps;
  std::vector<std::unique_ptr<Evaluator>> evaluators;

  for (size_t i = 0; i < args.evaluator_factories.size(); ++i) {
    EvaluatorFactory* factory = args.evaluator_factories[i];
    const EvaluatorConfig& config = args.evaluator_configs[i];
    evaluator_caps.push_back(factory->get_capabilities());
    evaluators.emplace_back(factory->new_evaluator(config));
  }
  assert(evaluators.size() > 0);
  i32 last_evaluator_device_id = args.evaluator_configs.back().device_ids[0];
  DeviceType last_evaluator_device_type = evaluator_caps.back().device_type;

  for (auto& evaluator : evaluators) {
    evaluator->set_profiler(&args.profiler);
  }

  // Will be set when configure is called below
  std::vector<std::vector<std::string>> evaluator_output_columns;
  std::vector<i32> num_evaluator_outputs;
  i32 last_evaluator_num_columns;

  args.profiler.add_interval("setup", setup_start, now());

  i32 last_table_id = -1;
  i32 last_end_row = -1;
  i32 last_item_id = -1;
  while (true) {
    auto idle_start = now();
    // Wait for next work item to process
    EvalWorkEntry work_entry;
    args.input_work.pop(work_entry);

    if (work_entry.io_item_index == -1) {
      break;
    }

    LOG(INFO) << "Evaluate (N/PU/G: " << rank << "/" << args.id << "/"
              << args.evaluator_group << "): processing item "
              << work_entry.io_item_index;

    args.profiler.add_interval("idle", idle_start, now());

    auto work_start = now();

    const IOItem& io_item = args.io_items[work_entry.io_item_index];
    const BatchConfig& batch_config = args.metadata.at(io_item.table_id);

    // Make the evaluator aware of the format of the data
    if (work_entry.needs_configure) {
      // Thread new set of columns through evaluators
      evaluator_output_columns.clear();
      num_evaluator_outputs.clear();

      BatchConfig bc = batch_config;
      bc.input_columns = work_entry.column_names;
      for (size_t i = 0; i < evaluators.size(); ++i) {
        auto& evaluator = evaluators[i];
        EvaluatorFactory* factory = args.evaluator_factories[i];

        evaluator->configure(bc);

        // TODO(apoms): move this outside as a preamble that just runs through
        //  every input column configurations
        bc.input_columns = factory->get_output_columns(bc.input_columns);
        evaluator_output_columns.push_back(bc.input_columns);
        num_evaluator_outputs.push_back(evaluator_output_columns.back().size());
      }
      last_evaluator_num_columns = num_evaluator_outputs.back();
    }
    if (work_entry.needs_reset) {
      for (auto& evaluator : evaluators) {
        evaluator->reset();
      }
    }

    EvalWorkEntry output_work_entry;
    output_work_entry.io_item_index = work_entry.io_item_index;
    output_work_entry.buffer_handle = {
        evaluator_caps.back().device_type,
        args.evaluator_configs.back().device_ids[0]};
    output_work_entry.needs_configure = work_entry.needs_configure;
    output_work_entry.needs_reset = work_entry.needs_reset;
    output_work_entry.last_in_io_item = work_entry.last_in_io_item;
    output_work_entry.column_names = evaluator_output_columns.back();

    BatchedColumns& work_item_output_columns = output_work_entry.columns;
    work_item_output_columns.resize(last_evaluator_num_columns);

    i32 current_input = 0;
    i32 total_inputs = 0;
    for (size_t i = 0; i < work_entry.columns.size(); ++i) {
      total_inputs =
          std::max(total_inputs, (i32)work_entry.columns[i].rows.size());
    }
    while (current_input < total_inputs) {
      i32 batch_size =
          std::min(total_inputs - current_input, (i32)WORK_ITEM_SIZE);

      std::vector<std::string> input_names;
      BatchedColumns input_columns;
      DeviceHandle input_handle;
      // Initialize the output buffers with the frame input because we
      // perform a swap from output to input on each iterator to pass outputs
      // from the previous evaluator into the input of the next one
      std::vector<std::string> output_names = work_entry.column_names;
      BatchedColumns output_columns;
      output_columns.resize(work_entry.columns.size());
      for (size_t i = 0; i < work_entry.columns.size(); ++i) {
        i32 batch =
            std::min(batch_size, (i32)work_entry.columns[i].rows.size());
        assert(batch > 0);
        output_columns[i].rows.insert(
            output_columns[i].rows.end(),
            work_entry.columns[i].rows.begin() + current_input,
            work_entry.columns[i].rows.begin() + current_input + batch);
      }
      DeviceHandle output_handle = work_entry.buffer_handle;
      for (size_t e = 0; e < evaluators.size(); ++e) {
        EvaluatorCapabilities& caps = evaluator_caps[e];
        i32 device_id = args.evaluator_configs[e].device_ids[0];
        DeviceHandle current_handle = {caps.device_type, device_id};
        std::unique_ptr<Evaluator>& evaluator = evaluators[e];
        i32 num_outputs = num_evaluator_outputs[e];

        input_names.swap(output_names);
        input_columns.swap(output_columns);
        input_handle = output_handle;

        i32 num_inputs = input_columns.size();
        // If current evaluator type and input buffer type differ, then move
        // the data in the input buffer into a new buffer which has the same
        // type as the evaluator input
        auto copy_start = now();
        if (current_handle != input_handle) {
          for (i32 i = 0; i < num_inputs; ++i) {
            std::vector<u8*> dest_buffers, src_buffers;
            std::vector<size_t> sizes;

            Column& column = input_columns[i];
            size_t total_size = 0;
            for (i32 b = 0; b < (i32)column.rows.size(); ++b) {
              total_size += column.rows[b].size;
            }

            if (column.rows.size() > 0) {
              u8* block = new_block_buffer({caps.device_type, device_id},
                                           total_size,
                                           column.rows.size());
              for (i32 b = 0; b < (i32)column.rows.size(); ++b) {
                size_t size = column.rows[b].size;
                dest_buffers.push_back(block);
                block += size;
                src_buffers.push_back(column.rows[b].buffer);
                sizes.push_back(size);
              }

              auto memcpy_start = now();
              memcpy_vec(dest_buffers, current_handle, src_buffers,
                         input_handle, sizes);
              args.profiler.add_interval("memcpy", memcpy_start, now());

              auto delete_start = now();
              for (i32 b = 0; b < (i32)column.rows.size(); ++b) {
                delete_buffer(input_handle, column.rows[b].buffer);
                column.rows[b].buffer = dest_buffers[b];
              }
            }
          }

          input_handle = current_handle;
        }
        args.profiler.add_interval("evaluator_marshal", copy_start, now());

        // Setup output buffers to receive evaluator output
        output_columns.clear();
        output_handle = current_handle;
        output_columns.resize(num_outputs);
        output_names = evaluator_output_columns.at(e);

        auto eval_start = now();
        evaluator->evaluate(input_columns, output_columns);
        args.profiler.add_interval("evaluate", eval_start, now());
        // Do not verify outputs == inputs if we are decoding encoded video as
        // there is an increase of 1 encoded chunk to multiple frames
        if (false) {
          for (size_t i = 0; i < output_columns.size(); ++i) {
            LOG_IF(FATAL, output_columns[i].rows.size() != batch_size)
                << "Evaluator " << e << " produced "
                << output_columns[i].rows.size() << " output rows for column "
                << output_names[i] << ". Expected " << batch_size
                << " outputs.";
          }
        }
        // Allow passing input buffers through to an evaluator output
        // by tracking the pointers and comparing the output pointers
        // for equality
        std::set<u8*> all_output_buffers_set;
        for (Column& column : output_columns) {
          for (Row& row : column.rows) {
            all_output_buffers_set.insert(row.buffer);
          }
        }

        // Delete input buffers after they are used
        for (size_t i = 0; i < num_inputs; ++i) {
          Column& column = input_columns[i];
          for (Row& row : column.rows) {
            u8* buff = row.buffer;
            if (all_output_buffers_set.count(buff) == 0) {
              delete_buffer(input_handle, buff);
            }
          }
        }
      }
      for (i32 i = 0; i < last_evaluator_num_columns; ++i) {
        i32 num_output_rows = static_cast<i32>(output_columns[i].rows.size());
        work_item_output_columns[i].rows.insert(
            work_item_output_columns[i].rows.end(),
            output_columns[i].rows.begin(), output_columns[i].rows.end());
      }
      current_input += batch_size;
    }

    args.profiler.add_interval("task", work_start, now());

    LOG(INFO) << "Evaluate (N/PU/G: " << rank << "/" << args.id << "/"
              << args.evaluator_group << "): finished item "
              << work_entry.io_item_index;

    args.output_work.push(output_work_entry);
  }

  LOG(INFO) << "Evaluate (N/PU: " << rank << "/" << args.id
            << "): thread finished";

  THREAD_RETURN_SUCCESS();
}

void* post_evaluate_thread(void* arg) {
  PostEvaluateThreadArgs& args =
      *reinterpret_cast<PostEvaluateThreadArgs*>(arg);

  i32 rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  EvalWorkEntry buffered_entry;
  i64 current_offset = 0;
  while (true) {
    auto idle_start = now();
    // Wait for next work item to process
    EvalWorkEntry work_entry;
    args.input_work.pop(work_entry);

    if (work_entry.io_item_index == -1) {
      break;
    }

    LOG(INFO) << "Post-evaluate (N/PU: " << rank << "/" << args.id << "): "
              << "processing item " << work_entry.io_item_index;

    args.profiler.add_interval("idle", idle_start, now());

    auto work_start = now();

    const IOItem& io_item = args.io_items[work_entry.io_item_index];
    const BatchConfig& batch_config = args.metadata.at(io_item.table_id);

    if (buffered_entry.columns.size() == 0) {
      buffered_entry.columns.resize(work_entry.columns.size());
      buffered_entry.column_names = work_entry.column_names;
      buffered_entry.io_item_index = work_entry.io_item_index;
      buffered_entry.buffer_handle = work_entry.buffer_handle;
    }

    i64 num_rows = work_entry.columns[0].rows.size();
    i32 warmup_frames;
    if (work_entry.needs_reset) {
      i32 total_warmup_frames =
          std::min((i64)args.warmup_count, io_item.start_row);
      warmup_frames =
          std::min(num_rows, std::max(0L, total_warmup_frames - current_offset));
    } else {
      warmup_frames = 0;
    }
    current_offset += num_rows;
    for (size_t i = 0; i < work_entry.columns.size(); ++i) {
      // Delete warmup frame outputs
      for (i32 w = 0; w < warmup_frames; ++w) {
        delete_buffer(work_entry.buffer_handle,
                      work_entry.columns[i].rows[w].buffer);
      }
      // Keep non-warmup frame outputs
      buffered_entry.columns[i].rows.insert(
          buffered_entry.columns[i].rows.end(),
          work_entry.columns[i].rows.begin() + warmup_frames,
          work_entry.columns[i].rows.end());
    }

    if (work_entry.last_in_io_item) {
      args.output_work.push(buffered_entry);
      buffered_entry.columns.clear();
      buffered_entry.column_names.clear();
    }
  }
  THREAD_RETURN_SUCCESS();
}

}
