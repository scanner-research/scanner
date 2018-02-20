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

#include "scanner/api/enumerator.h"
#include "scanner/engine/enumerator_factory.h"
#include "scanner/engine/enumerator_registry.h"
#include "scanner/util/memory.h"

namespace scanner {
namespace internal {

EnumeratorRegistration::EnumeratorRegistration(
    const EnumeratorBuilder& builder) {
  const std::string& name = builder.name_;
  EnumeratorConstructor constructor = builder.constructor_;
  internal::EnumeratorFactory* factory =
      new internal::EnumeratorFactory(name, constructor);
  internal::EnumeratorRegistry* registry = internal::get_enumerator_registry();
  registry->add_enumerator(name, factory);
}

}
}
