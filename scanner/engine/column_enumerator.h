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

#include "scanner/api/enumerator.h"

#include "storehouse/storage_backend.h"
#include "scanner/engine/video_index_entry.h"
#include "scanner/engine/table_meta_cache.h"

#include <glog/logging.h>
#include <vector>

namespace scanner {
namespace internal {

class ColumnEnumerator : public Enumerator {
 public:
  ColumnEnumerator(const EnumeratorConfig& config);

  i64 total_elements() override;

  ElementArgs element_args_at(i64 element_idx) override;

  void set_table_meta(TableMetaCache* cache);

 private:
  Result valid_;
  std::string table_name_;
  std::string column_name_;
  TableMetaCache* table_metadata_;  // Caching table metadata
  i32 table_id_;
  i32 column_id_;

  i64 total_rows_;
};

}
}  // namespace scanner
