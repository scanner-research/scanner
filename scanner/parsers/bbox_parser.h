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

#include "scanner/server/results_parser.h"

#include <string>
#include <vector>

namespace scanner {

class BBoxParser : public ResultsParser {
 public:
  BBoxParser(const std::vector<std::string>& column_names);

  std::vector<std::string> get_output_names() override;

  void configure(const DatasetItemMetadata& metadata) override;

  void parse_output(const std::vector<u8*>& output,
                    const std::vector<i64>& output_size,
                    folly::dynamic& parsed_results) override;

 protected:
  std::vector<std::string> column_names_;
  DatasetItemMetadata metadata_;
};
}
