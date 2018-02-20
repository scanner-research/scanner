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

#include "scanner/api/enumerator.h"
#include "scanner/util/common.h"

#include <vector>

namespace scanner {
namespace internal {

/**
 * @brief Interface for constructing enumerators at runtime.
 */
class EnumeratorFactory {
 public:
  EnumeratorFactory(const std::string& enumerator_name,
                    EnumeratorConstructor constructor)
    : name_(enumerator_name),
      constructor_(constructor) {}

  const std::string& get_name() const { return name_; }

  /* @brief Constructs a kernel to be used for processing elements of data.
   */
  Enumerator* new_instance(const EnumeratorConfig& config) {
    return constructor_(config);
  }

 private:
  std::string name_;
  EnumeratorConstructor constructor_;
};

}
}
