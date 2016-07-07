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

#include <string>

namespace lightscan {

class DatabaseConfig {
 public:
  virtual ~DatabaseConfig() {};

  static DatabaseConfig *make_sql_config(
    const std::string &unix_socket,
    const std::string &db_name,
    const std::string &user,
    const std::string &password);

  static DatabaseConfig *make_disk_config(
    const std::string &root_directory,
    const std::string &database_name);

  virtual std::string name() = 0;
};

}
