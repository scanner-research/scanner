#include "scanner/engine/evaluate_worker.h"

#include "scanner/engine/evaluator_registry.h"
#include "scanner/video/decoder_automata.h"

#include <thread>

namespace scanner {
namespace internal {
namespace {
void move_if_different_address_space(Profiler &profiler,
                                     DeviceHandle current_handle,
                                     DeviceHandle target_handle,
                                     RowList &column) {
  if (!current_handle.is_same_address_space(target_handle)) {
    std::vector<u8 *> dest_buffers, src_buffers;
    std::vector<size_t> sizes;

    size_t total_size = 0;
    for (i32 b = 0; b < (i32)column.rows.size(); ++b) {
      total_size += column.rows[b].size;
    }

    if (column.rows.size() > 0) {
      u8 *block =
          new_block_buffer(target_handle, total_size, column.rows.size());
      for (i32 b = 0; b < (i32)column.rows.size(); ++b) {
        size_t size = column.rows[b].size;
        dest_buffers.push_back(block);
        block += size;
        src_buffers.push_back(column.rows[b].buffer);
        sizes.push_back(size);
      }

      auto memcpy_start = now();
      memcpy_vec(dest_buffers, target_handle, src_buffers, current_handle,
                 sizes);
      profiler.add_interval("memcpy", memcpy_start, now());

      auto delete_start = now();
      for (i32 b = 0; b < (i32)column.rows.size(); ++b) {
        delete_buffer(current_handle, column.rows[b].buffer);
        column.rows[b].buffer = dest_buffers[b];
      }
    }
  }
}

void move_if_different_address_space(Profiler &profiler,
                                     DeviceHandle current_handle,
                                     DeviceHandle target_handle,
                                     BatchedColumns &columns) {
  for (i32 i = 0; i < (i32)columns.size(); ++i) {
    RowList &column = columns[i];
    move_if_different_address_space(profiler, current_handle, target_handle,
                                    column);
  }
}
}

void *pre_evaluate_thread(void *arg) {
  PreEvaluateThreadArgs &args = *reinterpret_cast<PreEvaluateThreadArgs *>(arg);

  i64 work_item_size = rows_per_work_item();

  i32 last_table_id = -1;
  i32 last_end_row = -1;
  i32 last_item_id = -1;

  DeviceHandle decoder_output_handle;
  std::vector<std::unique_ptr<DecoderAutomata>> decoders;
  while (true) {
    auto idle_start = now();
    // Wait for next work item to process
    EvalWorkEntry work_entry;
    args.input_work.pop(work_entry);

    if (work_entry.io_item_index == -1) {
      break;
    }

    LOG(INFO) << "Pre-evaluate (N/KI: " << args.node_id << "/" << args.id
              << "): "
              << "processing item " << work_entry.io_item_index;

    args.profiler.add_interval("idle", idle_start, now());

    auto work_start = now();

    const IOItem &io_item = args.io_items[work_entry.io_item_index];

    bool needs_configure = !(io_item.table_id == last_table_id);
    bool needs_reset = needs_configure ||
                       !(io_item.item_id == last_item_id ||
                         (io_item.table_id == last_table_id &&
                          io_item.start_row == last_end_row));

    last_table_id = io_item.table_id;
    last_end_row = io_item.end_row;
    last_item_id = io_item.item_id;

    // Split up a work entry into work item size chunks
    i64 total_rows = io_item.end_row - io_item.start_row;

    i64 r = 0;
    if (!needs_reset) {
      i32 total_warmup_frames =
          std::min((i64)args.warmup_count, io_item.start_row);
      r = total_warmup_frames;
    }

    if (needs_configure) {
      decoders.clear();
    }
    // Setup decoders if they have not been initialized yet
    if (decoders.empty()) {
      VideoDecoderType decoder_type;
      i32 num_devices;
      // Select a decoder type based on the type of the first evaluator and
      // the available decoders
      if (args.device_handle.type == DeviceType::GPU &&
          VideoDecoder::has_decoder_type(VideoDecoderType::NVIDIA)) {
        decoder_output_handle.type = DeviceType::GPU;
        decoder_output_handle.id = args.device_handle.id;
        decoder_type = VideoDecoderType::NVIDIA;
        num_devices = 1;
      } else {
        decoder_output_handle = CPU_DEVICE;
        decoder_type = VideoDecoderType::SOFTWARE;
        num_devices = args.num_cpus;
      }
      for (size_t c = 0; c < work_entry.columns.size(); ++c) {
        if (work_entry.column_types[c] == ColumnType::Video) {
          decoders.emplace_back(new DecoderAutomata(args.device_handle,
                                                    num_devices, decoder_type));
        }
      }
    }

    i32 media_col_idx = 0;
    std::vector<std::vector<proto::DecodeArgs>> decode_args;
    bool first_item = true;
    std::vector<EvalWorkEntry> work_items;
    for (size_t c = 0; c < work_entry.columns.size(); ++c) {
      if (work_entry.column_types[c] == ColumnType::Video) {
        decode_args.emplace_back();
        auto &args = decode_args.back();
        for (Row row : work_entry.columns[c].rows) {
          args.emplace_back();
          proto::DecodeArgs &da = args.back();
          da.ParseFromArray(row.buffer, row.size);
        }
        decoders[media_col_idx]->initialize(args);
        media_col_idx++;
      } else {
        move_if_different_address_space(args.profiler, work_entry.buffer_handle,
                                        decoder_output_handle,
                                        work_entry.columns[c]);
      }
    }

    for (; r < total_rows; r += work_item_size) {
      media_col_idx = 0;
      EvalWorkEntry entry;
      entry.io_item_index = work_entry.io_item_index;
      entry.buffer_handle = decoder_output_handle;
      entry.needs_configure = first_item ? needs_configure : false;
      entry.needs_reset = first_item ? needs_reset : false;
      entry.last_in_io_item = (r + work_item_size >= total_rows) ? true : false;
      entry.columns.resize(work_entry.columns.size());
      for (size_t c = 0; c < work_entry.columns.size(); ++c) {
        i64 start = r;
        i64 end = std::min(r + work_item_size, total_rows);
        if (work_entry.column_types[c] == ColumnType::Video) {
          // Perform decoding
          i64 num_rows = end - start;
          size_t frame_size = decode_args[media_col_idx][0].width() *
                              decode_args[media_col_idx][0].height() * 3;
          u8 *buffer = new_block_buffer(decoder_output_handle,
                                        num_rows * frame_size, num_rows);
          decoders[media_col_idx]->get_frames(buffer, num_rows);
          for (i64 n = 0; n < num_rows; ++n) {
            INSERT_ROW(entry.columns[c], buffer + frame_size * n, frame_size);
          }
          media_col_idx++;
        } else {
          entry.columns[c].rows =
              std::vector<Row>(work_entry.columns[c].rows.begin() + start,
                               work_entry.columns[c].rows.begin() + end);
        }
      }
      // Push entry to kernels
      args.output_work.push(entry);
      first_item = false;
    }
  }
  LOG(INFO) << "Pre-evaluate (N/PU: " << args.node_id << "/" << args.id
            << "): thread finished ";
  THREAD_RETURN_SUCCESS();
}

void *evaluate_thread(void *arg) {
  EvaluateThreadArgs &args = *reinterpret_cast<EvaluateThreadArgs *>(arg);

  auto setup_start = now();

  // Instantiate kernels
  const std::vector<std::vector<i32>> &dead_columns = args.dead_columns;
  const std::vector<std::vector<i32>> &unused_outputs = args.unused_outputs;
  const std::vector<std::vector<i32>> &column_mapping = args.column_mapping;
  std::vector<DeviceHandle> kernel_devices;
  std::vector<i32> kernel_num_outputs;
  std::vector<std::unique_ptr<Kernel>> kernels;
  {
    EvaluatorRegistry *registry = get_evaluator_registry();
    for (size_t i = 0; i < args.kernel_factories.size(); ++i) {
      KernelFactory *factory = std::get<0>(args.kernel_factories[i]);
      const Kernel::Config &config = std::get<1>(args.kernel_factories[i]);
      kernel_devices.push_back(config.devices[0]);
      kernel_num_outputs.push_back(
          registry->get_evaluator_info(factory->get_evaluator_name())
              ->output_columns()
              .size());
      kernels.emplace_back(factory->new_instance(config));
    }
  }
  assert(kernels.size() > 0);
  i32 num_final_output_columns = kernel_num_outputs.back();

  for (auto &kernel : kernels) {
    kernel->set_profiler(&args.profiler);
  }

  args.profiler.add_interval("setup", setup_start, now());

  while (true) {
    auto idle_start = now();
    // Wait for next work item to process
    EvalWorkEntry work_entry;
    args.input_work.pop(work_entry);

    if (work_entry.io_item_index == -1) {
      break;
    }

    LOG(INFO) << "Evaluate (N/KI/G: " << args.node_id << "/" << args.ki << "/"
              << args.kg << "): processing item " << work_entry.io_item_index;

    args.profiler.add_interval("idle", idle_start, now());

    auto work_start = now();

    const IOItem &io_item = args.io_items[work_entry.io_item_index];

    // Make the evaluator aware of the format of the data
    if (work_entry.needs_reset) {
      for (auto &kernel : kernels) {
        kernel->reset();
      }
    }

    EvalWorkEntry output_work_entry;
    output_work_entry.io_item_index = work_entry.io_item_index;
    output_work_entry.buffer_handle = kernel_devices.back();
    output_work_entry.needs_configure = work_entry.needs_configure;
    output_work_entry.needs_reset = work_entry.needs_reset;
    output_work_entry.last_in_io_item = work_entry.last_in_io_item;

    BatchedColumns &work_item_output_columns = output_work_entry.columns;
    work_item_output_columns.resize(num_final_output_columns);

    i32 current_input = 0;
    i32 total_inputs = 0;
    for (size_t i = 0; i < work_entry.columns.size(); ++i) {
      total_inputs = // io_item.end_row - io_item.start_row;
          std::max(total_inputs, (i32)work_entry.columns[i].rows.size());
    }
    while (current_input < total_inputs) {
      i32 batch_size =
          std::min(total_inputs - current_input, (i32)WORK_ITEM_SIZE);

      BatchedColumns side_input_columns;
      DeviceHandle input_handle;
      // Initialize the output buffers with the frame input because we
      // perform a swap from output to input on each iterator to pass outputs
      // from the previous evaluator into the input of the next one
      BatchedColumns side_output_columns;
      side_output_columns.resize(work_entry.columns.size());
      for (size_t i = 0; i < work_entry.columns.size(); ++i) {
        i32 batch =
            std::min(batch_size, (i32)work_entry.columns[i].rows.size());
        assert(batch > 0);
        side_output_columns[i].rows.insert(
            side_output_columns[i].rows.end(),
            work_entry.columns[i].rows.begin() + current_input,
            work_entry.columns[i].rows.begin() + current_input + batch);
      }
      DeviceHandle output_handle = work_entry.buffer_handle;
      for (size_t k = 0; k < kernels.size(); ++k) {
        DeviceHandle current_handle = kernel_devices[k];
        std::unique_ptr<Kernel> &kernel = kernels[k];
        i32 num_outputs = kernel_num_outputs[k];

        DeviceHandle input_handle = output_handle;
        // Map from previous output columns to the set of input columns needed
        // by the kernel
        BatchedColumns input_columns;
        for (i32 in_col_idx : column_mapping[k]) {
          assert(in_col_idx < side_output_columns.size());
          input_columns.push_back(side_output_columns[in_col_idx]);
        }
        // If current evaluator type and input buffer type differ, then move
        // the data in the input buffer into a new buffer which has the same
        // type as the evaluator input
        auto copy_start = now();
        move_if_different_address_space(args.profiler, input_handle,
                                        current_handle, input_columns);

        input_handle = current_handle;
        args.profiler.add_interval("evaluator_marshal", copy_start, now());

        // Setup output buffers to receive evaluator output
        DeviceHandle output_handle = current_handle;
        BatchedColumns output_columns;
        output_columns.resize(num_outputs);

        auto eval_start = now();
        kernel->execute(input_columns, output_columns);
        args.profiler.add_interval("evaluate", eval_start, now());
        // Delete unused outputs
        for (size_t y = 0; y < unused_outputs[k].size(); ++y) {
          i32 unused_col_idx = unused_outputs[k][unused_outputs.size() - 1 - y];
          RowList &column = output_columns[unused_col_idx];
          for (Row &row : column.rows) {
            u8 *buff = row.buffer;
            delete_buffer(output_handle, buff);
          }
          output_columns.erase(output_columns.begin() + unused_col_idx);
        }
        // Verify the kernel produced the correct amount of output
        for (size_t i = 0; i < output_columns.size(); ++i) {
          LOG_IF(FATAL, output_columns[i].rows.size() != batch_size)
              << "Evaluator " << k << " produced "
              << output_columns[i].rows.size() << " output rows for column "
              << i << ". Expected " << batch_size << " outputs.";
        }
        // Delete dead columns
        for (size_t y = 0; y < dead_columns[k].size(); ++y) {
          i32 dead_col_idx = dead_columns[k][dead_columns[k].size() - 1 - y];
          RowList &column = side_output_columns[dead_col_idx];
          for (Row &row : column.rows) {
            u8 *buff = row.buffer;
            delete_buffer(output_handle, buff);
          }
          side_output_columns.erase(side_output_columns.begin() + dead_col_idx);
        }
        // Add new output columns
        for (const RowList &column : output_columns) {
          side_output_columns.push_back(column);
        }
      }
      assert(num_final_output_columns == side_output_columns.size());
      for (i32 i = 0; i < num_final_output_columns; ++i) {
        i32 num_output_rows =
            static_cast<i32>(side_output_columns[i].rows.size());
        work_item_output_columns[i].rows.insert(
            work_item_output_columns[i].rows.end(),
            side_output_columns[i].rows.begin(),
            side_output_columns[i].rows.end());
      }
      current_input += batch_size;
    }

    args.profiler.add_interval("task", work_start, now());

    LOG(INFO) << "Evaluate (N/KI/G: " << args.node_id << "/" << args.ki << "/"
              << args.kg << "): finished item " << work_entry.io_item_index;

    args.output_work.push(output_work_entry);
  }

  LOG(INFO) << "Evaluate (N/KI: " << args.node_id << "/" << args.ki
            << "): thread finished";

  THREAD_RETURN_SUCCESS();
}

void *post_evaluate_thread(void *arg) {
  PostEvaluateThreadArgs &args =
      *reinterpret_cast<PostEvaluateThreadArgs *>(arg);

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

    LOG(INFO) << "Post-evaluate (N/PU: " << args.node_id << "/" << args.id
              << "): processing item " << work_entry.io_item_index;

    args.profiler.add_interval("idle", idle_start, now());

    auto work_start = now();

    const IOItem &io_item = args.io_items[work_entry.io_item_index];

    if (buffered_entry.columns.size() == 0) {
      buffered_entry.io_item_index = work_entry.io_item_index;
      buffered_entry.columns.resize(work_entry.columns.size());
      buffered_entry.buffer_handle = work_entry.buffer_handle;
    }

    i64 num_rows = work_entry.columns[0].rows.size();
    i32 warmup_frames;
    if (work_entry.needs_reset) {
      i32 total_warmup_frames =
          std::min((i64)args.warmup_count, io_item.start_row);
      warmup_frames = std::min(
          num_rows, std::max(0L, total_warmup_frames - current_offset));
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
    }
  }

  LOG(INFO) << "Post-evaluate (N/PU: " << args.node_id << "/" << args.id
            << "): thread finished ";

  THREAD_RETURN_SUCCESS();
}
}
}
