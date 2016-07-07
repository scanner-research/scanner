/* Copyright 2016 Carnegie Mellon University, NVIDIA Corporation
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

#include "vale/storage/disk_storage.h"
#include "vale/util/util.h"

#include <libgen.h>

namespace vale {
namespace internal {

////////////////////////////////////////////////////////////////////////////////
/// DiskRandomReadFile
class DiskRandomReadFile : public RandomReadFile {
public:
  DiskRandomReadFile(
    const std::string& file_path)
    : file_path_(file_path)
  {
    log_vale.debug("DiskRandomReadFile: opening %s for reading",
                  file_path.c_str());
    fp_ = fopen(file_path.c_str(), "r");
    if (fp_ == NULL) {
      log_vale.fatal("FATAL: could not open %s for reading", file_path.c_str());
      exit(EXIT_FAILURE);
    }
    position_ = 0;
  }

  ~DiskRandomReadFile() {
    if (fp_ != NULL) {
      fclose(fp_);
    }
  }

  StoreResult read(
    uint64_t offset,
    size_t size,
    char* data,
    size_t& size_read) override
  {
    if (position_ != offset) {
      if (fseek(fp_, offset, SEEK_SET) != 0) {
        if (ferror(fp_)) {
          log_vale.fatal("FATAL: Error in seek file %s to position %llu",
                        file_path_.c_str(), offset);
          exit(EXIT_FAILURE);
        }
      }
      position_ = offset;
    }

    size_read = fread(data, sizeof(char), size, fp_);
    position_ += size_read;

    if (ferror(fp_)) {
      log_vale.fatal("FATAL: Error in reading file %s at pos %llu, size %lu",
                    file_path_.c_str(), offset, size);
      exit(EXIT_FAILURE);
    }

    if (feof(fp_)) {
      return StoreResult::EndOfFile;
    } else {
      return StoreResult::Success;
    }
  }

private:
  const std::string file_path_;
  FILE* fp_;
  int position_;
};

////////////////////////////////////////////////////////////////////////////////
/// DiskWriteFile
class DiskWriteFile : public WriteFile {
public:
  DiskWriteFile(const std::string& file_path)
    : file_path_(file_path)
  {
    log_vale.debug("DiskWriteFile: opening %s for writing",
                  file_path.c_str());
    char* path;
    path = strdup(file_path.c_str());
    if (path == NULL) {
      log_vale.fatal("FATAL: could not strdup %s", file_path.c_str());
      exit(EXIT_FAILURE);
    }
    if (mkdir_p(dirname(path), S_IRWXU) != 0) {
      log_vale.fatal("FATAL: could not mkdir %s", path);
      exit(EXIT_FAILURE);
    }
    free(path);
    fp_ = fopen(file_path.c_str(), "w");
    if (fp_ == NULL) {
      log_vale.fatal("FATAL: could not open %s for writing", file_path.c_str());
      exit(EXIT_FAILURE);
    }
  }

  ~DiskWriteFile() {
    save();
    if (fp_ != NULL) {
      fclose(fp_);
    }
  }

  StoreResult append(size_t size, const char* data) override {
    size_t size_written = fwrite(data, sizeof(char), size, fp_);
    if (size_written != size) {
      log_vale.fatal("FATAL: Did not write all %lu bytes for file %s",
                    size, file_path_.c_str());
      exit(EXIT_FAILURE);
    }
    return StoreResult::Success;
  }

  StoreResult save() override {
    fflush(fp_);
    return StoreResult::Success;
  }

private:
  const std::string file_path_;
  FILE* fp_;
};

////////////////////////////////////////////////////////////////////////////////
/// DiskStorage
DiskStorage::DiskStorage(DiskConfig config)
  : data_directory_(config.data_directory)
{
}

DiskStorage::~DiskStorage() {
}

StoreResult DiskStorage::make_random_read_file(
  const std::string& name,
  RandomReadFile*& file)
{
  file = new DiskRandomReadFile(data_directory_ + "/" + name);
  return StoreResult::Success;
}

StoreResult DiskStorage::make_write_file(
  const std::string& name,
  WriteFile*& file)
{
  file = new DiskWriteFile(data_directory_ + "/" + name);
  return StoreResult::Success;
}

}
}
