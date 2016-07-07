# Copyright 2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(STORAGE_SOURCE_FILES
  storage_config.cpp
  storage_backend.cpp
  disk_storage.cpp
  gcs_storage.cpp)

set(DATABASE_SOURCE_FILES
  database_config.cpp
  database_backend.cpp
  disk_database.cpp)

add_library(storage OBJECT
  ${STORAGE_SOURCE_FILES}
  ${DATABASE_SOURCE_FILES})

add_dependencies(storage
  GoogleAPICppClient)
