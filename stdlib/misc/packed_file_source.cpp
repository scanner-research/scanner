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
#include "scanner/api/enumerator.h"
#include "stdlib/stdlib.pb.h"

#include "storehouse/storage_backend.h"
#include "scanner/engine/video_index_entry.h"
#include "scanner/engine/table_meta_cache.h"

#include <glog/logging.h>
#include <vector>

using storehouse::StorageBackend;
using storehouse::StorageConfig;
using storehouse::StoreResult;
using storehouse::WriteFile;
using storehouse::RandomReadFile;

namespace scanner {

class PackedFileEnumerator : public Enumerator {
 public:
  PackedFileEnumerator(const EnumeratorConfig& config) : Enumerator(config) {
    scanner::proto::PackedFileEnumeratorArgs args;
    bool parsed = args.ParseFromArray(config.args.data(), config.args.size());
    if (!parsed) {
      RESULT_ERROR(&valid_, "Could not parse PackedFileEnumeratorArgs");
      return;
    }
    if (args.storage_type() == "") {
      args.set_storage_type("posix");
    }

    // Setup storagebackend using config arguments
    StorageConfig* sc_config = nullptr;
    if (args.storage_type() == "posix") {
      sc_config = StorageConfig::make_posix_config();
    } else if (args.storage_type() == "gcs" || args.storage_type() == "aws") {
      sc_config = StorageConfig::make_s3_config(args.bucket(), args.region(),
                                                args.endpoint());
    } else {
      LOG(FATAL) << "Not a valid storage config type";
    }
    storage_.reset(storehouse::StorageBackend::make_from_config(sc_config));
    assert(storage_.get());

    path_ = args.path();
  }

  i64 total_elements() override {
    init();

    return element_offsets_.size() - 1;
  }

  ElementArgs element_args_at(i64 element_idx) override {
    init();

    proto::PackedFileElementArgs args;
    args.set_path(path_);
    args.set_offset(element_offsets_[element_idx]);
    args.set_size(element_offsets_[element_idx + 1] -
                  element_offsets_[element_idx]);
    size_t size = args.ByteSizeLong();

    ElementArgs element_args;
    element_args.args.resize(size);
    args.SerializeToArray(element_args.args.data(), size);
    element_args.row_id = element_idx;

    return element_args;
  }

 private:
  void init() {
    if (!init_) {
      // Read file and determine number of rows
      std::unique_ptr<RandomReadFile> file;
      BACKOFF_FAIL(make_unique_random_read_file(
          storage_.get(),
          path_, file));

      u64 file_size = 0;
      BACKOFF_FAIL(file->get_size(file_size));

      // Read number of elements in file
      u64 pos = 0;
      std::vector<u64> element_sizes;
      u64 num_elements = 0;

      u64 elements = s_read<u64>(file.get(), pos);

      // Read element sizes from work item file header
      size_t prev_size = element_sizes.size();
      element_sizes.resize(prev_size + elements);
      s_read(file.get(),
             reinterpret_cast<u8*>(element_sizes.data() + prev_size),
             elements * sizeof(u64), pos);

      u64 offset = sizeof(u64) * (1 + elements);
      for (u64 size : element_sizes) {
        element_offsets_.push_back(offset);
        offset += size;
      }
      element_offsets_.push_back(offset);
    }
    init_ = true;
  }

  Result valid_;
  std::string path_;
  std::unique_ptr<storehouse::StorageBackend>
      storage_;  // Setup a distinct storage backend for each IO thread

  bool init_ = false;
  std::vector<u64> element_offsets_;
};

class PackedFileSource : public Source {
 public:
  PackedFileSource(const SourceConfig& config) :
      Source(config) {
    scanner::proto::PackedFileSourceArgs args;
    if (config.args.size() == 0) {
      // Sane defaults
      args.set_storage_type("posix");
    } else {
      bool parsed = args.ParseFromArray(config.args.data(), config.args.size());
      if (!parsed) {
        RESULT_ERROR(&valid_, "Could not parse ColumnSourceArgs");
        return;
      }
    }

    // Setup storagebackend using config arguments
    StorageConfig* sc_config = nullptr;
    if (args.storage_type() == "posix") {
      sc_config = StorageConfig::make_posix_config();
    } else if (args.storage_type() == "gcs" || args.storage_type() == "aws") {
      sc_config = StorageConfig::make_s3_config(args.bucket(), args.region(),
                                                args.endpoint());
    } else {
      LOG(FATAL) << "Not a valid storage config type";
    }
    storage_.reset(storehouse::StorageBackend::make_from_config(sc_config));
    assert(storage_.get());
  }

  void read(const std::vector<ElementArgs>& element_args,
            std::vector<Elements>& output_columns) override {
    // Deserialize all ElementArgs
    std::string path;
    std::vector<u64> offset_to_read;
    std::vector<u64> size_to_read;
    size_t total_size = 0;
    for (size_t i = 0; i < element_args.size(); ++i) {
      proto::PackedFileElementArgs a;
      bool parsed = a.ParseFromArray(element_args[i].args.data(),
                                     element_args[i].args.size());
      assert(parsed);
      LOG_IF(FATAL, !parsed) << "Could not parse element args in PackedFile";

      offset_to_read.push_back(a.offset());
      size_to_read.push_back(a.size());
      path = a.path();

      total_size += a.size();
    }

    // Allocate a buffer for all the data
    u8* block_buffer =
        new_block_buffer(CPU_DEVICE, total_size, size_to_read.size());

    // Read the data
    std::unique_ptr<RandomReadFile> file;
    if (element_args.size() > 0) {
      BACKOFF_FAIL(make_unique_random_read_file(storage_.get(), path, file));
    }
    u64 offset = 0;
    for (size_t i = 0; i < element_args.size(); ++i) {
      u8* dest_buffer = block_buffer + offset;
      u64 pos = offset_to_read[i];
      u64 size = size_to_read[i];
      s_read(file.get(), dest_buffer, size, pos);
      insert_element(output_columns[0], dest_buffer, size);

      offset += size;
    }
  }

 private:
  Result valid_;
  std::unique_ptr<storehouse::StorageBackend>
      storage_;  // Setup a distinct storage backend for each IO thread
};

REGISTER_ENUMERATOR(PackedFile, PackedFileEnumerator);

REGISTER_SOURCE(PackedFile, PackedFileSource).output("output");

}  // namespace scanner
