/* Copyright 2017 Carnegie Mellon University
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

#include "scanner/engine/table_meta_cache.h"

namespace scanner {
namespace internal {

TableMetaCache::TableMetaCache(storehouse::StorageBackend* storage,
                               const DatabaseMetadata& meta)
  : storage_(storage), meta_(meta) {}

const TableMetadata& TableMetaCache::at(const std::string& table_name) const {
  memoized_read(table_name);
  return cache_.at(table_name);
}

bool TableMetaCache::exists(const std::string& table_name) const {
  return meta_.has_table(table_name);
}

void TableMetaCache::update(const TableMetadata& meta) {
  cache_[meta.name()] = meta;
}

void TableMetaCache::memoized_read(const std::string& table_name) const {
  if (cache_.count(table_name) == 0 && meta_.has_table(table_name)) {
    std::string table_path =
        TableMetadata::descriptor_path(meta_.get_table_id(table_name));
    cache_.insert({table_name, read_table_metadata(storage_, table_path)});
  }
}

}
}
