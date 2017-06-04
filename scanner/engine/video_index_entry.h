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

#include "scanner/engine/metadata.h"
#include "scanner/engine/runtime.h"
#include "scanner/util/common.h"

#include "storehouse/storage_backend.h"

namespace scanner {
namespace internal {

struct VideoIndexEntry {
  i32 width;
  i32 height;
  i32 channels;
  FrameType frame_type;
  proto::VideoDescriptor::VideoCodecType codec_type;
  std::unique_ptr<storehouse::RandomReadFile> file;
  u64 file_size;
  std::vector<i64> keyframe_positions;
  std::vector<i64> keyframe_byte_offsets;
};

VideoIndexEntry read_video_index(storehouse::StorageBackend *storage,
                                 i32 table_id, i32 column_id, i32 item_id);

VideoIndexEntry read_video_index(storehouse::StorageBackend *storage,
                                 const VideoMetadata& video_meta);
}
}
