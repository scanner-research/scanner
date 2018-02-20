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

#include "scanner/engine/runtime.h"
#include "scanner/engine/source_factory.h"
#include "scanner/engine/table_meta_cache.h"
#include "scanner/util/common.h"
#include "scanner/util/queue.h"
#include "scanner/api/source.h"
#include "scanner/api/enumerator.h"

namespace scanner {
namespace internal {

struct LoadWorkerArgs {
  // Uniform arguments
  i32 node_id;
  TableMetaCache& table_meta;
  // Per worker arguments
  int worker_id;
  storehouse::StorageConfig* storage_config;
  Profiler& profiler;
  proto::Result& result;
  i32 io_packet_size;
  i32 work_packet_size;
  std::vector<SourceFactory*> source_factories;
  std::vector<SourceConfig> source_configs;
};

class LoadWorker {
 public:
  LoadWorker(const LoadWorkerArgs& args);

  void feed(LoadWorkEntry& input_entry);

  bool yield(i32 item_size, EvalWorkEntry& output_entry);

  bool done();

 private:
  const i32 node_id_;
  const i32 worker_id_;
  Profiler& profiler_;
  i32 io_packet_size_;
  i32 work_packet_size_;
  i32 num_columns_;
  std::vector<SourceConfig> source_configs_;
  std::vector<std::unique_ptr<Source>>
      sources_;  // Provides the implementation for reading
                 // data under the specified data sources

  // Continuation state
  bool first_item_;
  bool needs_configure_;
  bool needs_reset_;
  LoadWorkEntry entry_;
  i64 current_row_;
  i64 total_rows_;
};

}
}
