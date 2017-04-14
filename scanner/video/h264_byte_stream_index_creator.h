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

#include "scanner/api/database.h"
#include "scanner/util/common.h"
#include "scanner/util/h264.h"

#include "storehouse/storage_backend.h"
#include "storehouse/storage_config.h"

#include <string>

namespace scanner {
namespace internal {

class H264ByteStreamIndexCreator {
public:
  H264ByteStreamIndexCreator(storehouse::WriteFile* demuxed_bytestream);

  bool feed_packet(u8* data, size_t size);

  const std::vector<u8>& metadata_bytes() { return metadata_bytes_; }
  const std::vector<i64>& keyframe_positions() { return keyframe_positions_; }
  const std::vector<i64>& keyframe_timestamps() { return keyframe_timestamps_; }
  const std::vector<i64>& keyframe_byte_offsets() {
    return keyframe_byte_offsets_;
  };

  i32 frames() { return frame_; };
  i32 num_non_ref_frames() { return num_non_ref_frames_; };
  i32 nals_parsed() { return nals_parsed_; };

  std::string error_message() { return error_message_; }

 private:
  std::string error_message_;

  storehouse::WriteFile* demuxed_bytestream_;

  u64 bytestream_pos_ = 0;
  std::vector<u8> metadata_bytes_;
  std::vector<i64> keyframe_positions_;
  std::vector<i64> keyframe_timestamps_;
  std::vector<i64> keyframe_byte_offsets_;

  i64 frame_ = 0;
  bool in_meta_packet_sequence_ = false;
  i64 meta_packet_sequence_start_offset_ = 0;
  bool saw_sps_nal_ = false;
  bool saw_pps_nal_ = false;
  std::map<u32, SPS> sps_map_;
  std::map<u32, PPS> pps_map_;
  u32 last_sps_ = -1;
  u32 last_pps_ = -1;
  std::map<u32, std::vector<u8>> sps_nal_bytes_;
  std::map<u32, std::vector<u8>> pps_nal_bytes_;
  SliceHeader prev_sh_;

  i32 num_non_ref_frames_ = 0;
  i32 nals_parsed_ = 0;
};

}
}
