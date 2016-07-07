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

#include "lightscan/storage/database_backend.h"

#include <string>

namespace lightscan {

//////////////////////////////////////////////////////////////////////
/// DiskDatabaseConfig
struct DiskDatabaseConfig : public DatabaseConfig {
  std::string name() override { return database_name; }

  std::string root_directory;
  std::string database_name;
};

////////////////////////////////////////////////////////////////////////////////
/// DiskDatabase
class DiskDatabase : public DatabaseBackend {
public:
  DiskDatabase(DiskDatabaseConfig config);

  ~DiskDatabase();

  std::string name() override;

  bool has_table(const std::string &table) override;

  TableDescriptor fetch_table_descriptor(const std::string &table) override;

  bool insert_table(const TableDescriptor &descriptor) override;

  bool delete_table(const std::string &name) override;

protected:
  std::string root_table_directory_path();

  std::string table_directory_path(const std::string &table);

  std::string table_file_path(const std::string &table);

  std::string attribute_file_path(const std::string &table);

  const std::string root_directory_;
  const std::string database_name_;
  const std::string database_directory_;

private:
  void setup_database_directory();

};

}
