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
#include "scanner/util/thread_pool.h"

namespace scanner {
namespace internal {

static const i32 NUM_PREFETCH_THREADS = 16;

TableMetaCache::TableMetaCache(storehouse::StorageBackend* storage,
                               const DatabaseMetadata& meta)
  : storage_(storage), meta_(meta) {
  // Read table megafile

  std::string megafile_path = table_megafile_path();
  storehouse::FileInfo info;
  storehouse::StoreResult result;
  EXP_BACKOFF(storage_->get_file_info(megafile_path, info), result);
  if (result == storehouse::StoreResult::Success) {
    read_table_megafile(storage, cache_);
  }
}

const TableMetadata& TableMetaCache::at(const std::string& table_name) const {
  i32 table_id = meta_.get_table_id(table_name);
  memoized_read(table_id);
  std::lock_guard<std::mutex> lock(lock_);
  return cache_.at(table_id);
}

const TableMetadata& TableMetaCache::at(i32 table_id) const {
  memoized_read(table_id);
  std::lock_guard<std::mutex> lock(lock_);
  return cache_.at(table_id);
}

bool TableMetaCache::exists(const std::string& table_name) const {
  return meta_.has_table(table_name);
}

bool TableMetaCache::exists(i32 table_id) const {
  return meta_.has_table(table_id);
}

bool TableMetaCache::has(const std::string& table_name) const {
  i32 table_id = meta_.get_table_id(table_name);
  return cache_.count(table_id) > 0;
}

void TableMetaCache::update(const TableMetadata& meta) {
  std::lock_guard<std::mutex> lock(lock_);
  i32 table_id = meta_.get_table_id(meta.name());
  cache_[table_id] = meta;
}

void TableMetaCache::prefetch(const std::vector<std::string>& table_names) {
  VLOG(1) << "Prefetching table metadata";
  auto load_table_meta = [&](const std::string& table_name) {
    i32 table_id = meta_.get_table_id(table_name);
    if (meta_.table_is_committed(table_id)) {
      memoized_read(table_id);
    }
  };

  VLOG(1) << "Spawning thread pool";
  ThreadPool prefetch_pool(NUM_PREFETCH_THREADS);
  std::vector<std::future<void>> futures;
  for (const auto& t : table_names) {
    futures.emplace_back(prefetch_pool.enqueue(load_table_meta, t));
  }

  VLOG(1) << "Waiting on futures";
  for (auto& future : futures) {
    future.wait();
  }

  VLOG(1) << "Prefetch complete.";
}

void TableMetaCache::write_megafile() {
  write_table_megafile(storage_, cache_);
}


void TableMetaCache::memoized_read(const std::string& table_name) const {
  memoized_read(meta_.get_table_id(table_name));
}

void TableMetaCache::memoized_read(i32 table_id) const {
  bool b;
  {
    std::lock_guard<std::mutex> lock(lock_);
    b = cache_.count(table_id) == 0 && meta_.has_table(table_id);
  }
  if (b) {
    std::string table_path = TableMetadata::descriptor_path(table_id);
    TableMetadata meta = read_table_metadata(storage_, table_path);
    std::lock_guard<std::mutex> lock(lock_);
    cache_.insert({table_id, meta});
  }
}

}
}
