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
#include "scanner/util/storehouse.h"
#include "stdlib/stdlib.pb.h"

#include "storehouse/storage_backend.h"

#include <glog/logging.h>
#include <vector>

using storehouse::StorageBackend;
using storehouse::StorageConfig;
using storehouse::StoreResult;
using storehouse::RandomReadFile;

namespace scanner {

class FilesEnumerator : public Enumerator {
 public:
  FilesEnumerator(const EnumeratorConfig& config)
      : Enumerator(config) {
    scanner::proto::FilesEnumeratorArgs args;
    bool parsed = args.ParseFromArray(config.args.data(), config.args.size());
    if (!parsed) {
      RESULT_ERROR(&valid_, "Could not parse FilesEnumeratorArgs");
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

    paths_ = std::vector<std::string>(args.paths().begin(), args.paths().end());
  }

  i64 total_elements() override {
    return paths_.size();
  }

  ElementArgs element_args_at(i64 element_idx) override {
    proto::FilesElementArgs args;
    args.set_path(paths_.at(element_idx));
    size_t size = args.ByteSizeLong();

    ElementArgs element_args;
    element_args.args.resize(size);
    args.SerializeToArray(element_args.args.data(), size);
    element_args.row_id = element_idx;

    return element_args;
  }

 private:
  Result valid_;
  std::vector<std::string> paths_;

  std::unique_ptr<storehouse::StorageBackend>
      storage_;  // Setup a distinct storage backend for each IO thread
};

class FilesSource : public Source {
 public:
  FilesSource(const SourceConfig& config) :
      Source(config) {
    scanner::proto::FilesSourceArgs args;
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
    std::vector<std::string> paths;
    size_t total_size = 0;
    std::vector<size_t> sizes;
    for (size_t i = 0; i < element_args.size(); ++i) {
      proto::FilesElementArgs a;
      bool parsed = a.ParseFromArray(element_args[i].args.data(),
                                     element_args[i].args.size());
      assert(parsed);
      LOG_IF(FATAL, !parsed) << "Could not parse element args in FilesSource";

      paths.push_back(a.path());

      std::unique_ptr<RandomReadFile> file;
      BACKOFF_FAIL(make_unique_random_read_file(storage_.get(), a.path(), file));

      u64 file_size = 0;
      BACKOFF_FAIL(file->get_size(file_size));
      total_size += file_size;
      sizes.push_back(file_size);
    }

    // Allocate a buffer for all the data
    u8* block_buffer = new_block_buffer(CPU_DEVICE, total_size, paths.size());

    // Read the data
    u64 offset = 0;
    for (size_t i = 0; i < element_args.size(); ++i) {
      std::unique_ptr<RandomReadFile> file;
      BACKOFF_FAIL(make_unique_random_read_file(storage_.get(), paths[i], file));

      u8* dest_buffer = block_buffer + offset;
      u64 pos = 0;
      s_read(file.get(), dest_buffer, sizes[i], pos);

      insert_element(output_columns[0], dest_buffer, sizes[i]);

      offset += sizes[i];
    }
  }

 private:
  Result valid_;
  std::unique_ptr<storehouse::StorageBackend>
      storage_;  // Setup a distinct storage backend for each IO thread
};

REGISTER_ENUMERATOR(Files, FilesEnumerator);

REGISTER_SOURCE(Files, FilesSource).output("output");

}  // namespace scanner
