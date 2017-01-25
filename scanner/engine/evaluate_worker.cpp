#include "scanner/engine/evaluate_worker.h"

namespace scanner {
namespace internal {

void* pre_evaluate_thread(void* arg) {
  PreEvaluateThreadArgs& args = *reinterpret_cast<PreEvaluateThreadArgs*>(arg);

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

    std::vector<EvalWorkEntry> work_items;
    for (; r < total_rows; r += work_item_size) {
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
      printf("pushing item\n");
      args.output_work.push(output_work_entry);
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
