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

#pragma once

#include "scanner/util/common.h"

#include <folly/dynamic.h>

#include <vector>
#include <string>

namespace scanner {

class ResultsParser {
public:
  virtual std::vector<std::string> get_output_names() = 0;

  virtual void configure(const DatasetItemMetadata& metadata) = 0;

  virtual void parse_output(
    const std::vector<u8*>& output,
    const std::vector<i64>& output_size,
    folly::dynamic& parsed_results) = 0;
};

}
