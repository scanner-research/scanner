#include "scanner/engine/evaluate_worker.h"

#include "scanner/video/decoder_automata.h"

#include <thread>

namespace scanner {
namespace internal {
namespace {
void move_if_different_address_space(Profiler& profiler,
                                     DeviceHandle current_handle,
                                     DeviceHandle target_handle,
                                     RowList& column) {
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

void move_if_different_address_space(Profiler& profiler,
                                     DeviceHandle current_handle,
                                     DeviceHandle target_handle,
                                     BatchedColumns& columns) {
  for (i32 i = 0; i < (i32)columns.size(); ++i) {
    RowList &column = columns[i];
    move_if_different_address_space(profiler, current_handle, target_handle,
                                    column);
  }
}

}

void* pre_evaluate_thread(void* arg) {
  PreEvaluateThreadArgs& args = *reinterpret_cast<PreEvaluateThreadArgs*>(arg);

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

    const IOItem& io_item = args.io_items[work_entry.io_item_index];

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
        num_devices =
            (CPUS_PER_NODE != -1 ? CPUS_PER_NODE
                                 : std::thread::hardware_concurrency());
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
        auto& args = decode_args.back();
        for (Row row : work_entry.columns[c].rows) {
          args.emplace_back();
          proto::DecodeArgs& da = args.back();
          da.ParseFromArray(row.buffer, row.size);
        }
        decoders[media_col_idx]->initialize(args);
        media_col_idx++;
      }
    }
    for (; r < total_rows; r += work_item_size) {
      media_col_idx = 0;
      EvalWorkEntry entry;
      entry.io_item_index = work_entry.io_item_index;
      entry.buffer_handle = work_entry.buffer_handle;
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
          u8 *buffer = new_buffer(decoder_output_handle, num_rows * frame_size);
          printf("asking for %ld frames\n", num_rows);
          decoders[media_col_idx]->get_frames(buffer, num_rows);
          for (i64 n = 0; n < num_rows; ++n) {
            INSERT_ROW(entry.columns[c], buffer + frame_size * n, frame_size);
          }
          printf("got %ld frames\n", num_rows);
          media_col_idx++;
        } else {
          entry.columns[c].rows =
              std::vector<Row>(work_entry.columns[c].rows.begin() + start,
                               work_entry.columns[c].rows.begin() + end);
        }
      }
      // Push entry to kernels
      printf("pushing item\n");
      args.output_work.push(entry);
      first_item = false;
    }
  }
  THREAD_RETURN_SUCCESS();
}

void* evaluate_thread(void* arg) {
  EvaluateThreadArgs& args = *reinterpret_cast<EvaluateThreadArgs*>(arg);
}

void* post_evaluate_thread(void* arg) {
  PostEvaluateThreadArgs& args =
      *reinterpret_cast<PostEvaluateThreadArgs*>(arg);
}

}
}
