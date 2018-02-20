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

#include "scanner/api/source.h"

#include "storehouse/storage_backend.h"
#include "scanner/engine/video_index_entry.h"
#include "scanner/engine/table_meta_cache.h"

#include <glog/logging.h>
#include <vector>

namespace scanner {
namespace internal {

class ColumnSource : public Source {
 public:
  ColumnSource(const SourceConfig& config);

  void read(const std::vector<ElementArgs>& element_args,
            std::vector<Elements>& output_columns) override;

  void get_video_column_information(
      proto::VideoDescriptor::VideoCodecType& encoding_type,
      bool& inplace_video);

  void set_table_meta(TableMetaCache* cache);

 private:
  Result valid_;
  i32 load_sparsity_threshold_;
  TableMetaCache* table_metadata_;  // Caching table metadata
  std::unique_ptr<storehouse::StorageBackend>
      storage_;  // Setup a distinct storage backend for each IO thread

  // To ammortize opening files
  i32 last_table_id_ = -1;
  std::map<std::tuple<i32, i32, i32>, VideoIndexEntry> index_;

  // Video Column Information
  proto::VideoDescriptor::VideoCodecType codec_type_;
  bool inplace_video_;
};

}
}  // namespace scanner
