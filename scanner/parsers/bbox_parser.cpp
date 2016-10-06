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

#include "scanner/parsers/bbox_parser.h"

#include <fstream>
#include <queue>

namespace scanner {

BBoxParser::BBoxParser(const std::vector<std::string>& column_names)
    : column_names_(column_names) {}

std::vector<std::string> BBoxParser::get_output_names() {
  return column_names_;
}

void BBoxParser::configure(const DatasetItemMetadata& metadata) {
  metadata_ = metadata;
}

void BBoxParser::parse_output(const std::vector<u8*>& output,
                              const std::vector<i64>& output_size,
                              folly::dynamic& parsed_results) {
  size_t column_count = column_names_.size();
  for (size_t i = 0; i < column_count; ++i) {
    u8 *buf = output[i];
    size_t num_boxes = *((size_t *)buf);
    buf += sizeof(size_t);
    assert(output_size[i] == sizeof(size_t) + num_boxes * sizeof(BoundingBox));
    std::vector<BoundingBox> boxes(num_boxes);
    for (size_t i = 0; i < num_boxes; ++i) {
      boxes[i].x1 = *((f32 *)buf);
      buf += sizeof(f32);
      boxes[i].y1 = *((f32 *)buf);
      buf += sizeof(f32);
      boxes[i].x2 = *((f32 *)buf);
      buf += sizeof(f32);
      boxes[i].y2 = *((f32 *)buf);
      buf += sizeof(f32);
      boxes[i].confidence = *((f32 *)buf);
      buf += sizeof(f32);
    }

    folly::dynamic out_bboxes = folly::dynamic::array();
    for (auto &b : boxes) {
      folly::dynamic bbox = folly::dynamic::object();
      f32 width = b.x2 - b.x1;
      f32 height = b.y2 - b.y1;
      bbox["category"] = 0;
      bbox["x"] = b.x1 + width / 2;
      bbox["y"] = b.y1 + height / 2;
      bbox["width"] = width;
      bbox["height"] = height;
      bbox["confidence"] = b.confidence;
      out_bboxes.push_back(bbox);
    }

    parsed_results[column_names_[i]] = out_bboxes;
  }
}
}
