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

#pragma once

#include "lightscan/storage/storage_config.h"

#include <vector>
#include <string>
#include <memory>
#include <unistd.h>

namespace lightscan {

////////////////////////////////////////////////////////////////////////////////
/// StoreResult
enum class StoreResult {
  Success,
  EndOfFile,
  FileExists,
  FileDoesNotExist,
  TransientFailure,
};

std::string store_result_to_string(StoreResult result);

////////////////////////////////////////////////////////////////////////////////
/// RandomReadFile
class RandomReadFile {
public:
  virtual ~RandomReadFile() {};

  StoreResult read(
    uint64_t offset,
    size_t size,
    std::vector<char> &data);

  virtual StoreResult read(
    uint64_t offset,
    size_t size,
    char *data,
    size_t &size_read) = 0;
};

////////////////////////////////////////////////////////////////////////////////
/// WriteFile
class WriteFile {
public:
  virtual ~WriteFile() {};

  StoreResult append(const std::vector<char> &data);

  virtual StoreResult append(size_t size, const char *data) = 0;

  virtual StoreResult save() = 0;
};

////////////////////////////////////////////////////////////////////////////////
/// StorageBackend
class StorageBackend {
public:
  virtual ~StorageBackend() {}

  static StorageBackend *make_from_config(
    const StorageConfig *config);

  /* make_random_read_file
   *
   */
  virtual StoreResult make_random_read_file(
    const std::string &name,
    RandomReadFile *&file) = 0;

  /* make_write_file
   *
   */
  virtual StoreResult make_write_file(
    const std::string &name,
    WriteFile *&file) = 0;

};

////////////////////////////////////////////////////////////////////////////////
/// Utilities
StoreResult make_unique_random_read_file(
  StorageBackend *storage,
  const std::string &name,
  std::unique_ptr<RandomReadFile> &file);

StoreResult make_unique_write_file(
  StorageBackend *storage,
  const std::string &name,
  std::unique_ptr<WriteFile> &file);

void exit_on_error(StoreResult result);

#define EXP_BACKOFF(expression, status)                                 \
  do {                                                                  \
    int sleep_debt__ = 1;                                               \
    while (true) {                                                      \
      const StoreResult result__ = (expression);                        \
      if (result__ == StoreResult::TransientFailure) {                  \
        double sleep_time__ =                                           \
          (sleep_debt__ + (static_cast<double>(rand()) / RAND_MAX));    \
        if (sleep_debt__ < 64) {                                        \
          sleep_debt__ *= 2;                                            \
        } else {                                                        \
          log_ls.fatal("EXP_BACKOFF: FATAL, reached max backoff");    \
          exit(1);                                                      \
        }                                                               \
        log_ls.warning("EXP_BACKOFF: transient failure, sleeping %fs\n", \
                         sleep_time__);                                 \
        usleep(sleep_time__ * 1000000);                                   \
        continue;                                                       \
      }                                                                 \
      status = result__;                                                \
      break;                                                            \
    }                                                                   \
  } while (0)


#define RETURN_ON_ERROR(expression)             \
  do {                                          \
    const StoreResult result = (expression);    \
    if (result != StoresResult::Success) {      \
      return result;                            \
    }                                           \
  } while (0)

}
