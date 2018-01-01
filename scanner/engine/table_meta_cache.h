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

#pragma once

#include "scanner/engine/metadata.h"

#include <map>
#include <mutex>

namespace scanner {
namespace internal {

class TableMetaCache {
 public:
  TableMetaCache(storehouse::StorageBackend* storage,
                 const DatabaseMetadata& meta);

  const TableMetadata& at(const std::string& table_name) const;

  const TableMetadata& at(i32 table_id) const;

  bool exists(const std::string& table_name) const;

  bool exists(i32 table_id) const;

  void update(const TableMetadata& meta);

  void prefetch(const std::vector<std::string> table_names);

 private:
  void memoized_read(const std::string& table_name) const;

  void memoized_read(i32 table_id) const;

  storehouse::StorageBackend* storage_;
  const DatabaseMetadata& meta_;
  mutable std::mutex lock_;
  mutable std::map<i32, TableMetadata> cache_;
};

}
}
