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

#include "scanner/engine/video_index_entry.h"

namespace scanner {
namespace internal {

std::unique_ptr<storehouse::RandomReadFile> VideoIndexEntry::open_file() const {
  std::unique_ptr<storehouse::RandomReadFile> file;
  BACKOFF_FAIL(storehouse::make_unique_random_read_file(
      storage, table_item_output_path(table_id, column_id, item_id),
      file));
  return std::move(file);
}

VideoIndexEntry read_video_index(storehouse::StorageBackend* storage,
                                 i32 table_id, i32 column_id, i32 item_id) {
  VideoMetadata video_meta = read_video_metadata(
      storage, VideoMetadata::descriptor_path(table_id, column_id, item_id));
  return read_video_index(storage, video_meta);
}

VideoIndexEntry read_video_index(storehouse::StorageBackend* storage,
                                 const VideoMetadata& video_meta) {
  VideoIndexEntry index_entry;

  i32 table_id = video_meta.table_id();
  i32 column_id = video_meta.column_id();
  i32 item_id = video_meta.item_id();

  // Open the video file for reading
  index_entry.storage = storage;
  index_entry.table_id = table_id;
  index_entry.column_id = column_id;
  index_entry.item_id = item_id;
  index_entry.width = video_meta.width();
  index_entry.height = video_meta.height();
  index_entry.channels = video_meta.channels();
  index_entry.frame_type = video_meta.frame_type();
  index_entry.codec_type = video_meta.codec_type();

  std::unique_ptr<storehouse::RandomReadFile> file;
  BACKOFF_FAIL(storehouse::make_unique_random_read_file(
      storage, table_item_output_path(table_id, column_id, item_id),
      file));
  BACKOFF_FAIL(file->get_size(index_entry.file_size));
  index_entry.num_encoded_videos = video_meta.num_encoded_videos();
  index_entry.frames_per_video = video_meta.frames_per_video();
  index_entry.keyframes_per_video = video_meta.keyframes_per_video();
  index_entry.size_per_video = video_meta.size_per_video();
  index_entry.keyframe_positions = video_meta.keyframe_positions();
  index_entry.keyframe_byte_offsets = video_meta.keyframe_byte_offsets();
  if (index_entry.codec_type == proto::VideoDescriptor::H264) {
    // Update keyframe positions and byte offsets so that the separately
    // encoded videos seem like they are one
    i64 frame_offset = 0;
    i64 keyframe_offset = 0;
    i64 byte_offset = 0;
    for (i64 v = 0; v < index_entry.num_encoded_videos; ++v) {
      for (i64 i = 0; i < index_entry.keyframes_per_video[v]; ++i) {
        i64 fo = keyframe_offset + i;
        index_entry.keyframe_positions[fo] += frame_offset;
        index_entry.keyframe_byte_offsets[fo] += byte_offset;
      }
      frame_offset += index_entry.frames_per_video[v];
      keyframe_offset += index_entry.keyframes_per_video[v];
      byte_offset += index_entry.size_per_video[v];
    }

    // Place total frames at the end of keyframe positions and total file size
    // at the end of byte offsets to make interval calculation not need to
    // deal with edge cases surrounding those
    index_entry.keyframe_positions.push_back(video_meta.frames());
    index_entry.keyframe_byte_offsets.push_back(index_entry.file_size);
  }

  return index_entry;
}

}
}
