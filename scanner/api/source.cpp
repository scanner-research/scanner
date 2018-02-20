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

#include "scanner/api/source.h"
#include "scanner/engine/source_factory.h"
#include "scanner/engine/source_registry.h"
#include "scanner/util/memory.h"

namespace scanner {
namespace internal {

SourceRegistration::SourceRegistration(const SourceBuilder& builder) {
  const std::string& name = builder.name_;
  std::vector<Column> output_columns;
  i32 i = 0;
  for (auto& name_type : builder.output_columns_) {
    Column col;
    col.set_id(i++);
    col.set_name(std::get<0>(name_type));
    col.set_type(std::get<1>(name_type));
    output_columns.push_back(col);
  }
  SourceConstructor constructor = builder.constructor_;
  internal::SourceFactory* factory =
      new internal::SourceFactory(name, output_columns, constructor);
  internal::SourceRegistry* registry = internal::get_source_registry();
  registry->add_source(name, factory);
}

}
}
