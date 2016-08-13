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

#include "lightscan/storage/storage_backend.h"
#include "lightscan/storage/gcs_storage.h"
#include "lightscan/storage/disk_storage.h"
#include "lightscan/util/common.h"

#include <cstdlib>

namespace lightscan {

StoreResult RandomReadFile::read(
  uint64_t offset,
  size_t size,
  std::vector<char> &data)
{
  size_t orig_data_size = data.size();
  data.resize(orig_data_size + size);
  size_t size_read;
  StoreResult result =
    this->read(offset, size, data.data() + orig_data_size, size_read);
  data.resize(orig_data_size + size_read);
  return result;
}

StoreResult WriteFile::append(const std::vector<char> &data) {
  return this->append(data.size(), data.data());
}

StorageBackend *StorageBackend::make_from_config(
  const StorageConfig *config)
{
  if (const GCSConfig *gcs_config = dynamic_cast<const GCSConfig *>(config)) {
    return new GCSStorage(*gcs_config);
  } else if (const DiskConfig *d_config =
             dynamic_cast<const DiskConfig *>(config)) {
    return new DiskStorage(*d_config);
  }
  return nullptr;
}

std::string StoreResultToString(StoreResult result) {
  switch (result) {
  case StoreResult::Success:
    return "Success";
  case StoreResult::FileExists:
    return "FileExists";
  case StoreResult::FileDoesNotExist:
    return "FileDoesNotExist";
  case StoreResult::EndOfFile:
    return "EndOfFile";
  case StoreResult::TransientFailure:
    return "TransientFailure";
  }
  return "<Undefined>";
}

StoreResult make_unique_random_read_file(
  StorageBackend *storage,
  const std::string &name,
  std::unique_ptr<RandomReadFile> &file)
{
  RandomReadFile *ptr;

  int sleep_debt = 1;
  StoreResult result;
  // Exponential backoff for transient failures
  while (true) {
    result = storage->make_random_read_file(name, ptr);
    if (result == StoreResult::TransientFailure) {
      double sleep_time =
        (sleep_debt + (static_cast<double>(rand()) / RAND_MAX));
      if (sleep_debt < 64) {
        sleep_debt *= 2;
      } else {
        log_ls.fatal("GCSLoadIOIterator: FATAL, reached max backoff for %s",
                       name.c_str());
        exit(1);
      }
      log_ls.warning("GCSLoadIOIterator: transient failure %s, "
                       "sleeping %fs\n",
                       name.c_str(), sleep_time);
      usleep(sleep_time * 1000000);
      continue;
    }
    break;
  }
  file.reset(ptr);
  return result;
}

StoreResult make_unique_write_file(
  StorageBackend *storage,
  const std::string &name,
  std::unique_ptr<WriteFile> &file)
{
  WriteFile *ptr;

  int sleep_debt = 1;
  StoreResult result;
  // Exponential backoff for transient failures
  while (true) {
    result = storage->make_write_file(name, ptr);
    if (result == StoreResult::TransientFailure) {
      double sleep_time =
        (sleep_debt + (static_cast<double>(rand()) / RAND_MAX));
      if (sleep_debt < 64) {
        sleep_debt *= 2;
      } else {
        log_ls.fatal("make_unique_write_file: FATAL, reached max backoff "
                       "for %s",
                       name.c_str());
        exit(1);
      }
      log_ls.warning("make_unique_write_file: transient failure %s, "
                       "sleeping %fs\n",
                       name.c_str(), sleep_time);
      usleep(sleep_time * 1000000);
      continue;
    }
    break;
  }
  file.reset(ptr);
  return result;
}

std::vector<char> read_entire_file(RandomReadFile* file, uint64_t& pos) {
  // Load the entire input
  std::vector<char> bytes;
  {
    const size_t READ_SIZE = 1024 * 1024;
    while (true) {
      size_t prev_size = bytes.size();
      bytes.resize(bytes.size() + READ_SIZE);
      size_t size_read;
      StoreResult result;
      EXP_BACKOFF(
        file->read(pos, READ_SIZE, bytes.data() + prev_size, size_read),
        result);
      assert(result == StoreResult::Success ||
             result == StoreResult::EndOfFile);
      pos += size_read;
      if (result == StoreResult::EndOfFile) {
        bytes.resize(prev_size + size_read);
        break;
      }
    }
  }
  return bytes;
}

void exit_on_error(StoreResult result) {
  if (result == StoreResult::Success) return;

  log_ls.fatal("FATAL: Exiting due to StorageBackend operation result %s",
                StoreResultToString(result).c_str());
  std::exit(EXIT_FAILURE);
}

}
