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

#include "vale/storage/database_config.h"
#include "vale/common/types.h"

#include <vector>
#include <string>
#include <memory>

namespace vale {
namespace internal {

////////////////////////////////////////////////////////////////////////////////
/// DatabaseBackend
class DatabaseBackend {
public:
  virtual ~DatabaseBackend() {}

  static DatabaseBackend *make_from_config(
    const DatabaseConfig *config);

  virtual std::string name() = 0;

  virtual bool has_table(const std::string &table) = 0;

  virtual TableDescriptor fetch_table_descriptor(const std::string &table) = 0;

  virtual bool insert_table(const TableDescriptor &descriptor) = 0;

  virtual bool delete_table(const std::string &name) = 0;
};

}
}
