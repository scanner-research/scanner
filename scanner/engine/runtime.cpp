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

#include "scanner/api/run.h"
#include "scanner/engine/runtime.h"
#include "scanner/engine/evaluator_registry.h"
#include "scanner/engine/kernel_registry.h"
#include "scanner/engine/save_worker.h"
#include "scanner/engine/evaluate_worker.h"
#include "scanner/engine/load_worker.h"
#include "scanner/engine/db.h"
#include "scanner/engine/rpc.grpc.pb.h"

namespace scanner {

using namespace internal;

void run_job(JobParameters& params) {
  storehouse::StorageBackend* storage =
    storehouse::StorageBackend::make_from_config(params.storage_config);

  DatabaseMetadata db_meta = read_database_metadata(
    storage, DatabaseMetadata::descriptor_path());

  EvaluatorRegistry* evaluator_registry = get_evaluator_registry();
  KernelRegistry* kernel_registry = get_kernel_registry();
  std::vector<KernelFactory*> kernel_factories;
  std::vector<KernelConfig> kernel_configs;
  for (auto& evaluator : params.task_set.evaluators()) {
    const std::string& name = evaluator.name();
    KernelFactory* kernel_factory =
      kernel_registry->get_kernel(name, evaluator.device_type());
    kernel_factories.push_back(kernel_factory);

    KernelConfig kernel_config;
    kernel_config.args = std::vector<u8>(
      evaluator.kernel_args().begin(),
      evaluator.kernel_args().end());
    for (auto& input : evaluator.inputs()) {
      const proto::Evaluator& input_evaluator = params.task_set.evaluators(
        input.evaluator_index());
      EvaluatorInfo* evaluator_info =
        evaluator_registry->get_evaluator_info(input_evaluator.name());
      // TODO(wcrichto): verify that input.columns() are all in
      // evaluator_info->output_columns()
      kernel_config.input_columns.insert(
        kernel_config.input_columns.end(),
        input.columns().begin(),
        input.columns().end());
    }
    kernel_configs.push_back(kernel_config);
  }

  const i32 io_item_size = rows_per_io_item();
  const i32 work_item_size = rows_per_work_item();

  i32 warmup_size = 0;
  i32 num_nodes = 1;
  i32 total_rows = 0;
  std::vector<IOItem> io_items;
  std::vector<LoadWorkEntry> load_work_entries;
  std::vector<std::string> final_column_names;
  proto::JobDescriptor job_descriptor;
  job_descriptor.set_io_item_size(io_item_size);
  job_descriptor.set_work_item_size(work_item_size);
  job_descriptor.set_num_nodes(num_nodes);

  auto& tasks = params.task_set.tasks();
  job_descriptor.mutable_tasks()->CopyFrom(tasks);

  for (size_t i = 0; i < tasks.size(); ++i) {
    auto& task = tasks.Get(i);
    assert(task.samples().size() > 0);
    i64 item_id = 0;
    i64 rows_in_task = static_cast<i64>(task.samples(0).rows().size());
    i64 allocated_rows = 0;
    while (allocated_rows < rows_in_task) {
      i64 rows_to_allocate =
        std::min((i64)io_item_size, rows_in_task - allocated_rows);

      IOItem item;
      item.table_id = i;
      item.item_id = item_id++;
      item.start_row = allocated_rows;
      item.end_row = allocated_rows + rows_to_allocate;
      io_items.push_back(item);

      LoadWorkEntry load_item;
      load_item.io_item_index = io_items.size() - 1;
      for (auto& sample : task.samples()) {
        i32 sample_job_id = sample.job_id();
        i32 sample_table_id = sample.table_id();

        load_item.samples.emplace_back();
        LoadWorkEntry::Sample& load_sample = load_item.samples.back();
        load_sample.job_id = sample_job_id;
        load_sample.table_id = sample_table_id;
        // load_sample.columns.insert(
        //   load_sample.columns.begin(),
        //   sample.column_ids().begin(),
        //   sample.column_ids().end());
        i64 e = allocated_rows + rows_to_allocate;
        // Add extra frames for warmup
        i64 s = std::max(allocated_rows - warmup_size, 0L);
        for (; s < e; ++s) {
          load_sample.rows.push_back(sample.rows(s));
        }
      }
      load_work_entries.push_back(load_item);

      allocated_rows += rows_to_allocate;
    }
    total_rows += rows_in_task;
  }
}

}
