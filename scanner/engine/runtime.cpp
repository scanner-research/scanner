/* Copyright 2016 Carnegie Mellon University, NVIDIA Corporation
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

#include "scanner/engine/runtime.h"

#include <thread>

namespace scanner {
namespace internal {

void create_io_items(
  const proto::JobParameters* job_params,
  const std::map<std::string, TableMetadata> &table_metas,
  const proto::TaskSet &task_set,
  std::vector<IOItem> &io_items,
  std::vector<LoadWorkEntry> &load_work_entries,
  proto::Result* job_result) {
  const i32 io_item_size = job_params->io_item_size();
  auto &tasks = task_set.tasks();
  i32 warmup_size = 0;
  i32 total_rows = 0;
  for (size_t i = 0; i < tasks.size(); ++i) {
    auto &task = tasks.Get(i);
    if (table_metas.count(task.output_table_name()) == 0) {
      RESULT_ERROR(
        job_result,
        "Output table %s does not exist.", task.output_table_name().c_str());
      return;
    }
    i32 table_id = table_metas.at(task.output_table_name()).id();
    assert(task.samples().size() > 0);
    i64 item_id = 0;
    i64 rows_in_task = static_cast<i64>(task.samples(0).rows().size());
    i64 allocated_rows = 0;
    while (allocated_rows < rows_in_task) {
      i64 rows_to_allocate =
          std::min((i64)io_item_size, rows_in_task - allocated_rows);

      IOItem item;
      item.table_id = table_id;
      item.item_id = item_id++;
      item.start_row = allocated_rows;
      item.end_row = allocated_rows + rows_to_allocate;
      io_items.push_back(item);

      proto::LoadWorkEntry load_item;
      load_item.set_io_item_index(io_items.size() - 1);
      for (auto &sample : task.samples()) {
        if (table_metas.count(sample.table_name()) == 0) {
          RESULT_ERROR(
            job_result,
            "Requested table %s does not exist.", sample.table_name().c_str());
          return;
        }
        const TableMetadata &t_meta = table_metas.at(sample.table_name());
        i32 sample_table_id = t_meta.id();
        proto::LoadSample *load_sample = load_item.add_samples();
        load_sample->set_table_id(sample_table_id);
        for (auto col_name : sample.column_names()) {
          load_sample->add_column_ids(t_meta.column_id(col_name));
        }
        i64 e = allocated_rows + rows_to_allocate;
        // Add extra frames for warmup
        i64 s = std::max(allocated_rows - warmup_size, 0L);
        for (; s < e; ++s) {
          load_sample->add_rows(sample.rows(s));
        }
      }
      load_work_entries.push_back(load_item);

      allocated_rows += rows_to_allocate;
    }
    total_rows += rows_in_task;
  }
}

}
}
