#include "scanner/engine/evaluate_worker.h"

#include "scanner/engine/op_registry.h"
#include "scanner/util/cuda.h"
#include "scanner/video/decoder_automata.h"
#include "scanner/video/video_encoder.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <thread>

namespace scanner {
namespace internal {
namespace {
void move_if_different_address_space(Profiler& profiler,
                                     DeviceHandle current_handle,
                                     DeviceHandle target_handle,
                                     RowList& column) {
  if (!current_handle.is_same_address_space(target_handle)) {
    std::vector<u8*> dest_buffers, src_buffers;
    std::vector<size_t> sizes;

    size_t total_size = 0;
    for (i32 b = 0; b < (i32)column.rows.size(); ++b) {
      total_size += column.rows[b].size;
    }

    if (column.rows.size() > 0) {
      u8* block =
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
}

void move_if_different_address_space(Profiler& profiler,
                                     DeviceHandle current_handle,
                                     DeviceHandle target_handle,
                                     BatchedColumns& columns) {
  for (i32 i = 0; i < (i32)columns.size(); ++i) {
    RowList& column = columns[i];
    move_if_different_address_space(profiler, current_handle, target_handle,
                                    column);
  }
}

void* pre_evaluate_thread(void* arg) {
  PreEvaluateThreadArgs& args = *reinterpret_cast<PreEvaluateThreadArgs*>(arg);

  i64 work_item_size = args.job_params->work_item_size();

  i32 last_table_id = -1;
  i32 last_end_row = -1;
  i32 last_item_id = -1;

  DeviceHandle decoder_output_handle;
  std::vector<std::unique_ptr<DecoderAutomata>> decoders;
  while (true) {
    auto idle_start = now();
    // Wait for next work item to process

    std::tuple<IOItem, EvalWorkEntry> entry;
    args.input_work.pop(entry);
    IOItem& io_item = std::get<0>(entry);
    EvalWorkEntry& work_entry = std::get<1>(entry);
    if (work_entry.io_item_index == -1) {
      break;
    }

    VLOG(2) << "Pre-evaluate (N/KI: " << args.node_id << "/" << args.id << "): "
            << "processing item " << work_entry.io_item_index;

    args.profiler.add_interval("idle", idle_start, now());

    auto work_start = now();

    bool needs_configure = !(io_item.table_id() == last_table_id);
    bool needs_reset = true;
    // NOTE(apoms): for avoiding warmup
    // needs_configure || !(io_item.item_id() == last_item_id ||
    //       (io_item.table_id() == last_table_id &&
    //        io_item.start_row() == last_end_row));

    last_table_id = io_item.table_id();
    last_end_row = io_item.end_row();
    last_item_id = io_item.item_id();

    // Split up a work entry into work item size chunks
    i64 total_rows = io_item.end_row() - io_item.start_row();

    if (needs_configure) {
      // decoders.clear();
    }

    // Setup decoders if they have not been initialized yet
    if (decoders.empty()) {
      auto init_start = now();
      VideoDecoderType decoder_type;
      i32 num_devices;
      // Select a decoder type based on the type of the first op and
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
          decoders.back()->set_profiler(&args.profiler);
        }
      }
      args.profiler.add_interval("init", init_start, now());
    }

    i32 media_col_idx = 0;
    std::vector<std::vector<proto::DecodeArgs>> decode_args;
    bool first_item = true;
    std::vector<EvalWorkEntry> work_items;
    auto setup_start = now();
    for (size_t c = 0; c < work_entry.columns.size(); ++c) {
      if (work_entry.column_types[c] == ColumnType::Video) {
        decode_args.emplace_back();
        auto& args = decode_args.back();
        for (Row row : work_entry.columns[c].rows) {
          args.emplace_back();
          proto::DecodeArgs& da = args.back();
          google::protobuf::io::ArrayInputStream in_stream(row.buffer,
                                                           row.size);
          google::protobuf::io::CodedInputStream cstream(&in_stream);
          cstream.SetTotalBytesLimit(row.size + 1, row.size + 1);
          bool result = da.ParseFromCodedStream(&cstream);
          assert(result);
          delete_buffer(CPU_DEVICE, row.buffer);
        }
        decoders[media_col_idx]->initialize(args);
        media_col_idx++;
      }
    }
    args.profiler.add_interval("setup", setup_start, now());

    auto decode_start = now();
    for (i64 r = 0; r < total_rows; r += work_item_size) {
      media_col_idx = 0;
      EvalWorkEntry entry;
      entry.io_item_index = work_entry.io_item_index;
      entry.needs_configure = first_item ? needs_configure : false;
      entry.needs_reset = first_item ? needs_reset : false;
      entry.last_in_io_item = (r + work_item_size >= total_rows) ? true : false;
      entry.warmup_rows = work_entry.warmup_rows;
      entry.columns.resize(work_entry.columns.size());
      for (size_t c = 0; c < work_entry.columns.size(); ++c) {
        i64 start = r;
        i64 end = std::min(r + work_item_size, total_rows);
        if (work_entry.column_types[c] == ColumnType::Video) {
          // Perform decoding
          i64 num_rows = end - start;
          size_t frame_size = decode_args[media_col_idx][0].width() *
                              decode_args[media_col_idx][0].height() * 3;
          u8* buffer = new_block_buffer(decoder_output_handle,
                                        num_rows * frame_size, num_rows);
          decoders[media_col_idx]->get_frames(buffer, num_rows);
          for (i64 n = 0; n < num_rows; ++n) {
            INSERT_ROW(entry.columns[c], buffer + frame_size * n, frame_size);
          }
          entry.column_handles.push_back(decoder_output_handle);
          media_col_idx++;
        } else {
          entry.columns[c].rows =
              std::vector<Row>(work_entry.columns[c].rows.begin() + start,
                               work_entry.columns[c].rows.begin() + end);
          entry.column_handles.push_back(work_entry.column_handles[c]);
        }
      }
      // Push entry to kernels
      args.output_work.push(std::make_tuple(io_item, entry));
      first_item = false;
    }
    args.profiler.add_interval("decode", decode_start, now());
  }

  VLOG(1) << "Pre-evaluate (N/PU: " << args.node_id << "/" << args.id
          << "): thread finished ";
  THREAD_RETURN_SUCCESS();
}

void* evaluate_thread(void* arg) {
  EvaluateThreadArgs& args = *reinterpret_cast<EvaluateThreadArgs*>(arg);

  auto setup_start = now();

  // Instantiate kernels
  const std::vector<std::vector<i32>>& dead_columns = args.dead_columns;
  const std::vector<std::vector<i32>>& unused_outputs = args.unused_outputs;
  const std::vector<std::vector<i32>>& column_mapping = args.column_mapping;
  std::vector<DeviceHandle> kernel_devices;
  std::vector<i32> kernel_num_outputs;
  std::vector<std::unique_ptr<Kernel>> kernels;
  {
    OpRegistry* registry = get_op_registry();
    for (size_t i = 0; i < args.kernel_factories.size(); ++i) {
      KernelFactory* factory = std::get<0>(args.kernel_factories[i]);
      const Kernel::Config& config = std::get<1>(args.kernel_factories[i]);
      kernel_devices.push_back(config.devices[0]);
      kernel_num_outputs.push_back(registry->get_op_info(factory->get_op_name())
                                       ->output_columns()
                                       .size());

#ifdef HAVE_CUDA
      cudaSetDevice(0);
#endif
      auto kernel = factory->new_instance(config);
      kernel->validate(&args.result);
      VLOG(1) << "Kernel finished validation " << args.result.success();
      if (!args.result.success()) {
        VLOG(1) << "Kernel validate failed: " << args.result.msg();
        THREAD_RETURN_SUCCESS();
      }
      kernels.emplace_back(kernel);
    }
  }
  assert(kernels.size() > 0);

  for (auto& kernel : kernels) {
    kernel->set_profiler(&args.profiler);
  }

  args.profiler.add_interval("setup", setup_start, now());

  while (true) {
    auto idle_start = now();
    // Wait for next work item to process
    std::tuple<IOItem, EvalWorkEntry> entry;
    args.input_work.pop(entry);
    IOItem& io_item = std::get<0>(entry);
    EvalWorkEntry& work_entry = std::get<1>(entry);
    if (work_entry.io_item_index == -1) {
      break;
    }

    VLOG(2) << "Evaluate (N/KI/G: " << args.node_id << "/" << args.ki << "/"
            << args.kg << "): processing item " << work_entry.io_item_index;

    args.profiler.add_interval("idle", idle_start, now());

    auto work_start = now();

    // Make the op aware of the format of the data
    if (work_entry.needs_reset) {
      for (auto& kernel : kernels) {
        kernel->reset();
      }
    }

    EvalWorkEntry output_work_entry;
    output_work_entry.io_item_index = work_entry.io_item_index;
    output_work_entry.needs_configure = work_entry.needs_configure;
    output_work_entry.needs_reset = work_entry.needs_reset;
    output_work_entry.last_in_io_item = work_entry.last_in_io_item;
    output_work_entry.warmup_rows = work_entry.warmup_rows;

    BatchedColumns& work_item_output_columns = output_work_entry.columns;
    std::vector<DeviceHandle>& work_item_output_handles =
        output_work_entry.column_handles;
    i32 num_final_output_columns = 0;

    i32 current_input = 0;
    i32 total_inputs = 0;
    for (size_t i = 0; i < work_entry.columns.size(); ++i) {
      total_inputs =  // io_item.end_row - io_item.start_row;
          std::max(total_inputs, (i32)work_entry.columns[i].rows.size());
    }
    while (current_input < total_inputs) {
      i32 batch_size = std::min(total_inputs - current_input,
                                args.job_params->work_item_size());

      BatchedColumns side_input_columns;
      DeviceHandle input_handle;
      // Initialize the output buffers with the frame input because we
      // perform a swap from output to input on each iterator to pass outputs
      // from the previous op into the input of the next one
      std::vector<DeviceHandle> side_output_handles = work_entry.column_handles;
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
      for (size_t k = 0; k < kernels.size(); ++k) {
        DeviceHandle current_handle = kernel_devices[k];
        std::unique_ptr<Kernel>& kernel = kernels[k];
        i32 num_outputs = kernel_num_outputs[k];

        // Map from previous output columns to the set of input columns needed
        // by the kernel
        BatchedColumns input_columns;
        for (i32 in_col_idx : column_mapping[k]) {
          assert(in_col_idx < side_output_columns.size());

          // If current op type and input buffer type differ, then move
          // the data in the input buffer into a new buffer which has the same
          // type as the op input
          auto copy_start = now();
          move_if_different_address_space(
              args.profiler, side_output_handles[in_col_idx], current_handle,
              side_output_columns[in_col_idx]);
          side_output_handles[in_col_idx] = current_handle;

          input_handle = current_handle;
          args.profiler.add_interval("op_marshal", copy_start, now());

          input_columns.push_back(side_output_columns[in_col_idx]);
        }

        // Setup output buffers to receive op output
        DeviceHandle output_handle = current_handle;
        BatchedColumns output_columns;
        output_columns.resize(num_outputs);

        auto eval_start = now();
        kernel->execute(input_columns, output_columns);
        args.profiler.add_interval("evaluate", eval_start, now());
        // Delete unused outputs
        for (size_t y = 0; y < unused_outputs[k].size(); ++y) {
          i32 unused_col_idx =
              unused_outputs[k][unused_outputs[k].size() - 1 - y];
          RowList& column = output_columns[unused_col_idx];
          for (Row& row : column.rows) {
            u8* buff = row.buffer;
            delete_buffer(current_handle, buff);
          }
          output_columns.erase(output_columns.begin() + unused_col_idx);
        }
        // Verify the kernel produced the correct amount of output
        for (size_t i = 0; i < output_columns.size(); ++i) {
          LOG_IF(FATAL, output_columns[i].rows.size() != batch_size)
              << "Op " << k << " produced " << output_columns[i].rows.size()
              << " output rows for column " << i << ". Expected " << batch_size
              << " outputs.";
        }
        // Delete dead columns
        for (size_t y = 0; y < dead_columns[k].size(); ++y) {
          i32 dead_col_idx = dead_columns[k][dead_columns[k].size() - 1 - y];
          RowList& column = side_output_columns[dead_col_idx];
          for (Row& row : column.rows) {
            u8* buff = row.buffer;
            delete_buffer(side_output_handles[dead_col_idx], buff);
          }
          side_output_columns.erase(side_output_columns.begin() + dead_col_idx);
          side_output_handles.erase(side_output_handles.begin() + dead_col_idx);
        }
        // Add new output columns
        for (const RowList& column : output_columns) {
          side_output_columns.push_back(column);
          side_output_handles.push_back(current_handle);
        }
      }
      if (work_item_output_columns.size() == 0) {
        num_final_output_columns = side_output_columns.size();
        work_item_output_columns.resize(side_output_columns.size());
        work_item_output_handles = side_output_handles;
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

    VLOG(2) << "Evaluate (N/KI/G: " << args.node_id << "/" << args.ki << "/"
            << args.kg << "): finished item " << work_entry.io_item_index;

    args.output_work.push(std::make_tuple(io_item, output_work_entry));
  }

  VLOG(1) << "Evaluate (N/KI: " << args.node_id << "/" << args.ki
          << "): thread finished";

  THREAD_RETURN_SUCCESS();
}

void* post_evaluate_thread(void* arg) {
  PostEvaluateThreadArgs& args =
      *reinterpret_cast<PostEvaluateThreadArgs*>(arg);
  std::set<i32> column_set(args.column_mapping.begin(),
                           args.column_mapping.end());
  // Infer which columns are video columns by finding any column name with the
  // prefix "frame_info" and asserting the previous column is a video column
  std::vector<ColumnType> column_types;
  for (size_t i = 0; i < args.column_names.size() - 1; ++i) {
    const std::string info_prefix = "frame_info";
    const std::string& column_name = args.column_names[i + 1];
    ColumnType type = ColumnType::Other;
    if (column_name.compare(0, info_prefix.size(), info_prefix) == 0) {
      type = ColumnType::Video;
    }
    column_types.push_back(type);
  }
  column_types.push_back(ColumnType::Other);

  // Setup video encoders
  DeviceHandle encoder_handle = CPU_DEVICE;
  VideoEncoderType encoder_type = VideoEncoderType::SOFTWARE;
  std::vector<std::unique_ptr<VideoEncoder>> encoders;
  std::vector<bool> encoder_configured;
  for (ColumnType type : column_types) {
    if (type != ColumnType::Video) continue;
    encoders.emplace_back(
      VideoEncoder::make_from_config(encoder_handle, 1, encoder_type));
    encoder_configured.push_back(false);
  }

  EvalWorkEntry buffered_entry;
  i64 current_offset = 0;
  while (true) {
    auto idle_start = now();
    // Wait for next work item to process
    std::tuple<IOItem, EvalWorkEntry> entry;
    args.input_work.pop(entry);
    IOItem& io_item = std::get<0>(entry);
    EvalWorkEntry& work_entry = std::get<1>(entry);

    if (work_entry.io_item_index == -1) {
      break;
    }

    VLOG(2) << "Post-evaluate (N/PU: " << args.node_id << "/" << args.id
            << "): processing item " << work_entry.io_item_index;

    args.profiler.add_interval("idle", idle_start, now());

    auto work_start = now();

    // Setup row buffer if it was emptied
    if (buffered_entry.columns.size() == 0) {
      buffered_entry.io_item_index = work_entry.io_item_index;
      buffered_entry.columns.resize(args.column_mapping.size());
      buffered_entry.column_types = column_types;
      for (i32 col_idx : args.column_mapping) {
        buffered_entry.column_handles.push_back(
            work_entry.column_handles[col_idx]);
      }
      if (work_entry.needs_configure) {
        for (size_t i = 0; i < encoder_configured.size(); ++i) {
          encoder_configured[i] = false;
        }
      }
    }

    i64 num_rows = work_entry.columns[0].rows.size();
    i32 warmup_frames = work_entry.warmup_rows;
    current_offset += num_rows;

    i32 encoder_idx = 0;
    // Swizzle columns correctly
    for (size_t i = 0; i < args.column_mapping.size(); ++i) {
      i32 col_idx = args.column_mapping[i];
      ColumnType column_type = column_types[i];
      // Delete warmup frame outputs
      for (i32 w = 0; w < warmup_frames; ++w) {
        delete_buffer(work_entry.column_handles[col_idx],
                      work_entry.columns[col_idx].rows[w].buffer);
      }
      // Encode video frames
      if (column_type == ColumnType::Video) {
        auto start = work_entry.columns[col_idx].rows.begin() + warmup_frames;
        auto end = work_entry.columns[col_idx].rows.end();
        auto& encoder = encoders[encoder_idx];
        if (!encoder_configured[encoder_idx]) {
          // Configure encoder
          encoder_configured[encoder_idx] = true;
          // Read frame info column

          auto& rows = work_entry.columns[col_idx + 1].rows;
          u8* buffer = new_buffer(CPU_DEVICE, rows[0].size);
          memcpy_buffer((u8*)buffer, CPU_DEVICE, rows[0].buffer,
                        work_entry.column_handles[col_idx + 1], rows[0].size);
          FrameInfo frame_info;
          bool parsed = frame_info.ParseFromArray(buffer, rows[0].size);
          LOG_IF(FATAL, !parsed) << "Invalid frame info";
          delete_buffer(CPU_DEVICE, buffer);
          encoder->configure(frame_info);
        }

        // Pass frames into encoder
        for (auto r = start; r != end; ++r) {
          auto& row = *r;
          bool new_packet = encoder->feed(row.buffer, row.size);
          while (new_packet) {
            size_t buffer_size = 1 * 1024 * 1024;
            u8* buffer = new_buffer(CPU_DEVICE, buffer_size);
            size_t actual_size;
            new_packet = encoder->get_packet(buffer, buffer_size, actual_size);
            LOG_IF(FATAL, new_packet && actual_size > buffer_size)
              << "Packet buffer not large enough (" << buffer_size << " vs "
              << actual_size << ")";
            buffered_entry.columns[i].rows.push_back(Row{buffer, buffer_size});
          }
        }
        encoder_idx++;
      } else {
        // Keep non-warmup frame outputs
        buffered_entry.columns[i].rows.insert(
          buffered_entry.columns[i].rows.end(),
          work_entry.columns[col_idx].rows.begin() + warmup_frames,
          work_entry.columns[col_idx].rows.end());
      }
    }
    // Delete unused columns
    for (size_t i = 0; i < work_entry.columns.size(); ++i) {
      if (column_set.count(i) > 0) {
        continue;
      }
      for (i32 b = 0; b < work_entry.columns[i].rows.size(); ++b) {
        delete_buffer(work_entry.column_handles[i],
                      work_entry.columns[i].rows[b].buffer);
      }
    }

    encoder_idx = 0;
    // Flush row buffer
    if (work_entry.last_in_io_item) {
      // Flush video encoder and get rest of packets
      for (size_t i = 0; i < args.column_mapping.size(); ++i) {
        ColumnType column_type = column_types[i];
        if (column_type == ColumnType::Video) {
          auto& encoder = encoders[encoder_idx];

          // Get last packets in encoder
          bool new_packet = encoder->flush();
          while (new_packet) {
            size_t buffer_size = 1 * 1024 * 1024;
            u8* buffer = new_buffer(CPU_DEVICE, buffer_size);
            size_t actual_size;
            new_packet = encoder->get_packet(buffer, buffer_size, actual_size);
            LOG_IF(FATAL, new_packet && actual_size > buffer_size)
              << "Packet buffer not large enough (" << buffer_size << " vs "
              << actual_size << ")";
            buffered_entry.columns[i].rows.push_back(Row{buffer, buffer_size});
          }
          encoder_idx++;
        }
      }

      args.output_work.push(std::make_tuple(io_item, buffered_entry));
      buffered_entry.columns.clear();
    }
  }

  VLOG(1) << "Post-evaluate (N/PU: " << args.node_id << "/" << args.id
          << "): thread finished ";

  THREAD_RETURN_SUCCESS();
}
}
}
