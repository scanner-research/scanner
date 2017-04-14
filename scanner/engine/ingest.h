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

#include "scanner/api/database.h"
#include "scanner/util/common.h"

#include "storehouse/storage_backend.h"
#include "storehouse/storage_config.h"

#include <string>

namespace scanner {
namespace internal {

Result ingest_videos(storehouse::StorageConfig* storage_config,
                     const std::string& db_path,
                     const std::vector<std::string>& table_names,
                     const std::vector<std::string>& paths,
                     std::vector<FailedVideo>& failed_videos);

// void ingest_images(storehouse::StorageConfig *storage_config,
//                    const std::string &db_path, const std::string &table_name,
//                    const std::vector<std::string> &paths);
}
}
