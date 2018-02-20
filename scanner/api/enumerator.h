/* Copyright 2017 Carnegie Mellon University
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

#include "scanner/util/common.h"
#include "scanner/util/profiler.h"

#include <vector>

namespace scanner {

/**
 * @brief Parameters provided at instantiation of an enumerator.
 */
struct EnumeratorConfig {
  std::vector<u8> args;  //! Byte-string of proto args if given.
};

struct ElementArgs {
  i64 row_id;
  std::vector<u8> args;  //! Args
};

/**
 * @brief Interface for enumerating available data from a data source.
 */
class Enumerator {
 public:
  Enumerator(const EnumeratorConfig& config) {};
  virtual ~Enumerator() {};

  virtual void validate(proto::Result* result) { result->set_success(true); }

  virtual i64 total_elements() = 0;

  /**
   * @brief Returns the data that can be used by a Source to load the element at
   *        this index.
   */
  virtual ElementArgs element_args_at(i64 element_idx) = 0;
};

///////////////////////////////////////////////////////////////////////////////
/// Implementation Details
namespace internal {

class EnumeratorBuilder;

using EnumeratorConstructor =
    std::function<Enumerator*(const EnumeratorConfig& config)>;

class EnumeratorRegistration {
 public:
  EnumeratorRegistration(const EnumeratorBuilder& builder);
};

class EnumeratorBuilder {
 public:
  friend class EnumeratorRegistration;

  EnumeratorBuilder(const std::string& name, EnumeratorConstructor constructor)
    : name_(name),
      constructor_(constructor) {}

 private:
  std::string name_;
  EnumeratorConstructor constructor_;
};
}

#define REGISTER_ENUMERATOR(name__, enumerator__) \
  REGISTER_ENUMERATOR_HELPER(__COUNTER__, name__, enumerator__)

#define REGISTER_ENUMERATOR_HELPER(uid__, name__, enumerator__) \
  REGISTER_ENUMERATOR_UID(uid__, name__, enumerator__)

#define REGISTER_ENUMERATOR_UID(uid__, name__, enumerator__)           \
  static ::scanner::internal::EnumeratorRegistration                   \
      enumerator_registration_##uid__ __attribute__((unused)) =        \
          ::scanner::internal::EnumeratorBuilder(                      \
              #name__, [](const ::scanner::EnumeratorConfig& config) { \
                return new enumerator__(config);                       \
              })
}
