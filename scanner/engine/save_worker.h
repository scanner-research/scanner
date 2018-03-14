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

#include "scanner/api/sink.h"
#include "scanner/engine/runtime.h"
#include "scanner/engine/sink_factory.h"
#include "scanner/util/common.h"
#include "scanner/util/queue.h"
#include "scanner/util/storehouse.h"

namespace scanner {
namespace internal {

struct SaveWorkerArgs {
  // Uniform arguments
  i32 node_id;
  const std::vector<std::map<i32, std::vector<u8>>>& sink_args;

  // Per worker arguments
  int worker_id;
  storehouse::StorageConfig* storage_config;
  std::vector<SinkFactory*> sink_factories;
  std::vector<SinkConfig> sink_configs;
  Profiler& profiler;
  proto::Result& result;
};

class SaveWorker {
 public:
  SaveWorker(const SaveWorkerArgs& args);
  ~SaveWorker();

  void feed(EvalWorkEntry& input_entry);

  void new_task(i32 job_id, i32 task_id, i32 output_table_id,
                std::vector<ColumnType> column_types);

 private:
  const i32 node_id_;
  const i32 worker_id_;
  Profiler& profiler_;
  const std::vector<std::map<i32, std::vector<u8>>> sink_args_;

  //
  std::vector<i32> sink_op_idx_;
  std::vector<SinkConfig> sink_configs_;
  std::vector<std::unique_ptr<Sink>> sinks_;  // Provides the implementation for
                                              // writing data under the
                                              // specified data sources

  // Continuation state
  bool first_item_;
  bool needs_configure_;
  bool needs_reset_;

  i64 current_work_item_;
  i64 current_row_;
  i64 total_work_items_;

};

}
}
