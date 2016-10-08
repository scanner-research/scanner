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

class FacenetParser : public ResultsParser {
 public:
  FacenetParser(double threshold);

  std::vector<std::string> get_output_names() override;

  void configure(const VideoMetadata& metadata) override;

  void parse_output(const std::vector<u8*>& output,
                    const std::vector<i64>& output_size,
                    folly::dynamic& parsed_results) override;

 protected:
  struct Box {
    f32 x1;
    f32 y1;
    f32 x2;
    f32 y2;
    f32 confidence;
  };

  std::vector<Box> nms(std::vector<Box> boxes, f32 overlap);

  i32 num_templates_;
  i32 net_input_width_;
  i32 net_input_height_;
  i32 cell_width_;
  i32 cell_height_;
  i32 grid_width_;
  i32 grid_height_;
  std::vector<std::vector<f32>> templates_;
  std::vector<i32> feature_vector_lengths_;
  std::vector<size_t> feature_vector_sizes_;

  double threshold_;

  VideoMetadata metadata_;
};
}
