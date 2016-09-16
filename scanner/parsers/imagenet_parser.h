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

namespace scanner {

class ImagenetParser : public ResultsParser {
public:
  ImagenetParser();

  std::vector<std::string> get_output_names() override;

  void parse_output(
    const std::vector<u8*>& output,
    const std::vector<i64>& output_size,
    folly::dynamic& parsed_results) override;

private:
  static const size_t FEATURE_VECTOR_LENGTH = 1000;
  static const size_t FEATURE_VECTOR_SIZE = FEATURE_VECTOR_LENGTH * sizeof(f32);
};

}
