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

#include "scanner/parsers/imagenet_parser.h"

namespace scanner {

ImagenetParser::ImagenetParser() {}

std::vector<std::string> ImagenetParser::get_output_names() { return {"fc8"}; }

void ImagenetParser::parse_output(const std::vector<u8 *> &output,
                                  const std::vector<i64> &output_size,
                                  folly::dynamic &parsed_results) {
  assert(output_size[0] == FEATURE_VECTOR_SIZE);
  f32 *feature_vector = reinterpret_cast<f32 *>(output[0]);

  f64 norm = 0.0;
  for (size_t i = 0; i < FEATURE_VECTOR_LENGTH; ++i) {
    norm += feature_vector[i] * feature_vector[i];
  }

  f64 max_confidence = 0.0;
  i32 max_index = 0;

  int filtered_category = -1;
  if (filtered_category == -1) {
    for (size_t i = 0; i < FEATURE_VECTOR_LENGTH; ++i) {
      if (max_confidence < feature_vector[i] / norm) {
        max_confidence = feature_vector[i] / norm;
        max_index = static_cast<i32>(i);
      }
    }
  } else {
    max_confidence = feature_vector[filtered_category] / norm;
    max_index = filtered_category;
  }

  parsed_results["class"] = max_index;
  parsed_results["confidence"] = max_confidence;
}
}
