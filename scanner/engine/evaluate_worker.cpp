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

  DeviceType decoder_output_type;
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
    i64 total_rows = work_entry.columns[0].rows.size();

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
        decoder_output_type = DeviceType::GPU;
        decoder_type = VideoDecoderType::NVIDIA;
        num_devices = 1;
      } else {
        decoder_output_type = DeviceType::CPU;
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

    for (size_t c = 0; c < work_entry.columns.size(); ++c) {
    }

    bool first_item = true;
    std::vector<EvalWorkEntry> work_items;
    for (; r < total_rows; r += work_item_size) {
      EvalWorkEntry entry;
      entry.io_item_index = work_entry.io_item_index;
      entry.buffer_handle = work_entry.buffer_handle;
      entry.needs_configure = first_item ? needs_configure : false;
      entry.needs_reset = first_item ? needs_reset : false;
      entry.last_in_io_item = (r + work_item_size >= total_rows) ? true : false;
      entry.columns.resize(work_entry.columns.size());
      for (size_t c = 0; c < work_entry.columns.size(); ++c) {
        if (work_entry.column_types[c] == ColumnType::Video) {
          // Perform decoding
        } else {
          i64 start = std::min(r, (i64)work_entry.columns[c].rows.size());
          i64 end = std::min(r + work_item_size,
                             (i64)work_entry.columns[c].rows.size());
          entry.columns[c].rows =
              std::vector<Row>(work_entry.columns[c].rows.begin() + start,
                               work_entry.columns[c].rows.begin() + end);
        }
      }

      // Push entry to kernels
      args.output_work.push(entry);

      first_item = false;
    }
    assert(!work_items.empty());
    work_items.front().needs_configure = needs_configure;
    work_items.front().needs_reset = needs_reset;
    work_items.back().last_in_io_item = true;

    for (EvalWorkEntry& output_work_entry : work_items) {
      // Perform decoding
      for (size_t c = 0; c < work_entry.columns.size(); ++c) {
      }
      printf("pushing item\n");

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
