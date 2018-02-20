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

#pragma once

#include "scanner/api/source.h"
#include "scanner/engine/source_factory.h"

#include "scanner/util/common.h"

#include <map>

namespace scanner {
namespace internal {

class SourceRegistry {
 public:
  void add_source(const std::string& name, SourceFactory* factory);

  bool has_source(const std::string& name);

  SourceFactory* get_source(const std::string& name);

 private:
  std::map<std::string, SourceFactory*> factories_;
};

SourceRegistry* get_source_registry();
}
}
