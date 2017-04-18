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
    //        io_item.start_element() == last_end_element));

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
      if (work_entry.column_types[c] == ColumnType::Video &&
          work_entry.video_encoding_type[media_col_idx] ==
          proto::VideoDescriptor::H264) {
        decode_args.emplace_back();
        auto& args = decode_args.back();
        for (Element element : work_entry.columns[c]) {
          args.emplace_back();
          proto::DecodeArgs& da = args.back();
          google::protobuf::io::ArrayInputStream in_stream(element.buffer,
                                                           element.size);
          google::protobuf::io::CodedInputStream cstream(&in_stream);
          cstream.SetTotalBytesLimit(element.size + 1, element.size + 1);
          bool result = da.ParseFromCodedStream(&cstream);
          assert(result);
          delete_element(CPU_DEVICE, element);
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
          if (work_entry.video_encoding_type[media_col_idx] ==
              proto::VideoDescriptor::H264) {
            // Encoded as video
            FrameInfo frame_info(decode_args[media_col_idx][0].height(),
                                 decode_args[media_col_idx][0].width(), 3,
                                 FrameType::U8);
            u8* buffer = new_block_buffer(
              decoder_output_handle, num_rows * frame_info.size(), num_rows);
            decoders[media_col_idx]->get_frames(buffer, num_rows);
            for (i64 n = 0; n < num_rows; ++n) {
              insert_frame(
                entry.columns[c],
                new Frame(frame_info, buffer + frame_info.size() * n));
            }
            entry.column_handles.push_back(decoder_output_handle);
          } else {
            // Encoded as raw data
            FrameInfo frame_info = work_entry.frame_sizes[media_col_idx];
            for (i64 n = 0; n < num_rows; ++n) {
              Element& e = work_entry.columns[c][start + n];
              assert(e.size == frame_info.size());
              insert_frame(
                entry.columns[c],
                new Frame(frame_info, e.buffer));
            }
          }
          media_col_idx++;
        } else {
          entry.columns[c] =
              std::vector<Element>(work_entry.columns[c].begin() + start,
                               work_entry.columns[c].begin() + end);
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
          std::max(total_inputs, (i32)work_entry.columns[i].size());
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
            std::min(batch_size, (i32)work_entry.columns[i].size());
        assert(batch > 0);
        side_output_columns[i].insert(
            side_output_columns[i].end(),
            work_entry.columns[i].begin() + current_input,
            work_entry.columns[i].begin() + current_input + batch);
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
          ElementList& column = output_columns[unused_col_idx];
          for (Element& element : column) {
            delete_element(current_handle, element);
          }
          output_columns.erase(output_columns.begin() + unused_col_idx);
        }
        // Verify the kernel produced the correct amount of output
        for (size_t i = 0; i < output_columns.size(); ++i) {
          LOG_IF(FATAL, output_columns[i].size() != batch_size)
              << "Op " << k << " produced " << output_columns[i].size()
              << " output elements for column " << i << ". Expected " << batch_size
              << " outputs.";
        }
        // Delete dead columns
        for (size_t y = 0; y < dead_columns[k].size(); ++y) {
          i32 dead_col_idx = dead_columns[k][dead_columns[k].size() - 1 - y];
          ElementList& column = side_output_columns[dead_col_idx];
          for (Element& element : column) {
            delete_element(side_output_handles[dead_col_idx], element);
          }
          side_output_columns.erase(side_output_columns.begin() + dead_col_idx);
          side_output_handles.erase(side_output_handles.begin() + dead_col_idx);
        }
        // Add new output columns
        for (const ElementList& column : output_columns) {
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
        i32 num_output_elements =
            static_cast<i32>(side_output_columns[i].size());
        work_item_output_columns[i].insert(
            work_item_output_columns[i].end(),
            side_output_columns[i].begin(),
            side_output_columns[i].end());
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
  assert(args.column_mapping.size() == args.columns.size());

  std::set<i32> column_set(args.column_mapping.begin(),
                           args.column_mapping.end());

  // Setup video encoders
  // TODO(apoms): Make this dynamic based on the encoded column type
  DeviceHandle encoder_handle = CPU_DEVICE;
  VideoEncoderType encoder_type = VideoEncoderType::SOFTWARE;
  std::vector<std::unique_ptr<VideoEncoder>> encoders;
  std::vector<bool> encoder_configured;
  for (auto& col : args.columns) {
    ColumnType type = col.type();
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
      assert(work_entry.column_handles.size() == args.columns.size());
      for (size_t i = 0; i < args.columns.size(); ++i) {
        buffered_entry.column_types.push_back(args.columns[i].type());
        buffered_entry.column_handles.push_back(
          work_entry.column_handles[i]);
        if (args.columns[i].type() == ColumnType::Video) {
          assert(work_entry.columns[i].size() > 0);
          Frame* frame = work_entry.columns[i][0].as_frame();
          buffered_entry.frame_sizes.push_back(frame->as_frame_info());
        }
      }
      if (work_entry.needs_configure) {
        for (size_t i = 0; i < encoder_configured.size(); ++i) {
          encoder_configured[i] = false;
        }
      }
    }

    i64 num_rows = work_entry.columns[0].size();
    i32 warmup_frames = work_entry.warmup_rows;
    current_offset += num_rows;

    i32 encoder_idx = 0;
    // Swizzle columns correctly
    for (size_t i = 0; i < args.column_mapping.size(); ++i) {
      i32 col_idx = args.column_mapping[i];
      ColumnType column_type = args.columns[i].type();
      // Delete warmup frame outputs
      for (i32 w = 0; w < warmup_frames; ++w) {
        delete_element(work_entry.column_handles[col_idx],
                       work_entry.columns[col_idx][w]);
      }
      // Encode video frames
      if (column_type == ColumnType::Video &&
          buffered_entry.frame_sizes[encoder_idx].type == FrameType::U8) {
        {
          auto start = work_entry.columns[col_idx].begin();
          auto warmup_end =
            work_entry.columns[col_idx].begin() + warmup_frames;
          work_entry.columns[col_idx].erase(start, warmup_end);
        }
        auto& encoder = encoders[encoder_idx];
        if (!encoder_configured[encoder_idx]) {
          // Configure encoder
          encoder_configured[encoder_idx] = true;
          Frame* frame = work_entry.columns[col_idx][0].as_frame();
          encoder->configure(frame->as_frame_info());
        }

        // Move frames to device for the encoder
        move_if_different_address_space(args.profiler,
                                        work_entry.column_handles[col_idx],
                                        encoder_handle,
                                        work_entry.columns[col_idx]);

        // Pass frames into encoder
        for (auto& row : work_entry.columns[col_idx]) {
          Frame* frame = row.as_frame();
          bool new_packet = encoder->feed(frame->data, frame->size());
          while (new_packet) {
            size_t buffer_size = 1 * 1024 * 1024;
            u8* buffer = new_buffer(CPU_DEVICE, buffer_size);
            size_t actual_size;
            new_packet = encoder->get_packet(buffer, buffer_size, actual_size);
            LOG_IF(FATAL, new_packet && actual_size > buffer_size)
              << "Packet buffer not large enough (" << buffer_size << " vs "
              << actual_size << ")";
            insert_element(buffered_entry.columns[i], buffer, actual_size);
          }
        }
        encoder_idx++;
      } else {
        // Keep non-warmup frame outputs
        buffered_entry.columns[i].insert(
          buffered_entry.columns[i].end(),
          work_entry.columns[col_idx].begin() + warmup_frames,
          work_entry.columns[col_idx].end());
      }
    }
    // Delete unused columns
    for (size_t i = 0; i < work_entry.columns.size(); ++i) {
      if (column_set.count(i) > 0) {
        continue;
      }
      for (i32 b = 0; b < work_entry.columns[i].size(); ++b) {
        delete_element(work_entry.column_handles[i], work_entry.columns[i][b]);
      }
    }

    encoder_idx = 0;
    // Flush row buffer
    if (work_entry.last_in_io_item) {
      // Flush video encoder and get rest of packets
      for (size_t i = 0; i < args.column_mapping.size(); ++i) {
        ColumnType column_type = args.columns[i].type();
        if (column_type == ColumnType::Video &&
            buffered_entry.frame_sizes[encoder_idx].type == FrameType::U8) {
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
            // HACK(apoms): this is really hacky but we put the encoded data in
            // a frame so that we can communicate the frame size downstream
            insert_element(buffered_entry.columns[i], buffer, actual_size);
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
