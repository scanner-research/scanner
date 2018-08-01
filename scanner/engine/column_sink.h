/* Copyright 2018 Carnegie Mellon University
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

#include "scanner/api/sink.h"

#include "storehouse/storage_backend.h"
#include "scanner/engine/video_index_entry.h"
#include "scanner/engine/table_meta_cache.h"
#include "scanner/util/thread_pool.h"

#include <glog/logging.h>
#include <vector>

namespace scanner {
namespace internal {

class ColumnSink : public Sink {
 public:
  ColumnSink(const SinkConfig& config);

  ~ColumnSink();

  void new_stream(const std::vector<u8>& args) override;

  void write(const BatchedElements& input_columns) override;

  void new_task(i32 table_id, i32 task_id,
                std::vector<ColumnType> column_types);

  void finished();

  void provide_column_info(
      const std::vector<bool>& compressed,
      const std::vector<FrameInfo>& frame_info);

 private:
  Result valid_;
  ThreadPool thread_pool_;
  // Setup a distinct storage backend for each IO thread
  std::unique_ptr<storehouse::StorageBackend> storage_;
  // Files to write io packets to
  std::vector<std::unique_ptr<storehouse::WriteFile>> output_;
  std::vector<std::unique_ptr<storehouse::WriteFile>> output_metadata_;
  std::vector<VideoMetadata> video_metadata_;

  std::vector<ColumnType> column_types_;
  std::vector<bool> compressed_;
  std::vector<FrameInfo> frame_info_;

  // Continuation state
  bool first_item_;
  bool needs_configure_;
  bool needs_reset_;

  i64 current_work_item_;
  i64 current_row_;
  i64 total_work_items_;
};

}
}  // namespace scanner
