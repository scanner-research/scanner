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
#include "scanner/util/storehouse.h"
#include "stdlib/stdlib.pb.h"

#include "storehouse/storage_backend.h"

#include <glog/logging.h>
#include <vector>

using storehouse::StorageBackend;
using storehouse::StorageConfig;
using storehouse::StoreResult;
using storehouse::WriteFile;

namespace scanner {

class FilesSink : public Sink {
 public:
  FilesSink(const SinkConfig& config) :
      Sink(config) {
    scanner::proto::FilesSinkArgs args;
    if (config.args.size() == 0) {
      // Sane defaults
      args.set_storage_type("posix");
    } else {
      bool parsed = args.ParseFromArray(config.args.data(), config.args.size());
      if (!parsed) {
        RESULT_ERROR(&valid_, "Could not parse ColumnSinkArgs");
        return;
      }
    }

    // Setup storagebackend using config arguments
    StorageConfig* sc_config = nullptr;
    if (args.storage_type() == "posix") {
      sc_config = StorageConfig::make_posix_config();
    } else if (args.storage_type() == "gcs" || args.storage_type() == "s3") {
      sc_config = StorageConfig::make_s3_config(args.bucket(), args.region(),
                                                args.endpoint());
    } else {
      LOG(FATAL) << "Not a valid storage config type";
    }
    storage_.reset(storehouse::StorageBackend::make_from_config(sc_config));
    assert(storage_.get());
  }

  void new_stream(const std::vector<u8>& args) override {
    paths_.clear();

    scanner::proto::FilesSinkStreamArgs sargs;
    if (args.size() != 0) {
      bool parsed = sargs.ParseFromArray(args.data(), args.size());
      if (!parsed) {
        RESULT_ERROR(&valid_, "Could not parse FilesSinkStreamArgs");
        return;
      }
      paths_ =
          std::vector<std::string>(sargs.paths().begin(), sargs.paths().end());
    }
  }

  void write(const BatchedElements& input_columns) override {
    // Write the data
    for (size_t i = 0; i < input_columns[0].size(); ++i) {
      u64 offset = input_columns[0][i].index;
      assert(offset < paths_.size());
      std::unique_ptr<WriteFile> file;
      BACKOFF_FAIL(make_unique_write_file(
          storage_.get(), paths_.at(offset), file));

      s_write(file.get(), input_columns[0][i].buffer, input_columns[0][i].size);
    }
  }

 private:
  Result valid_;
  std::unique_ptr<storehouse::StorageBackend>
      storage_;  // Setup a distinct storage backend for each IO thread
  std::vector<std::string> paths_;
};

REGISTER_SINK(Files, FilesSink)
    .input("input")
    .per_element_output()
    .protobuf_name("FilesSinkArgs")
    .stream_protobuf_name("FilesSinkStreamArgs");
}  // namespace scanner
