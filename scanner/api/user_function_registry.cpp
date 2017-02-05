/* Copyright 2016 Carnegie Mellon University
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

#include "scanner/engine/user_function_registry.h"

namespace scanner {
namespace internal {

void UserFunctionRegistry::add_user_function(const std::string &name,
                                             const void *fn) {
  fns_.insert({name, fn});
}

bool UserFunctionRegistry::has_user_function(const std::string &name) {
  return fns_.count(name) > 0;
}

UserFunctionRegistry *get_user_function_registry() {
  static UserFunctionRegistry *registry = new UserFunctionRegistry;
  return registry;
}
}
}
