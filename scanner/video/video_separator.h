/* Copyright 2016 Carnegie Mellon University, NVIDIA Corporation
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
#include "scanner/util/queue.h"

#include <pthread.h>
#include <string>

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavformat/avio.h"
#include "libavutil/error.h"
#include "libswscale/swscale.h"
}

namespace scanner {

///////////////////////////////////////////////////////////////////////////////
/// VideoSeparator
class VideoSeparator {
 public:
  VideoSeparator(AVCodecContext* cc);

  ~VideoSeparator();

  bool decode(AVPacket* packet);

  CUVIDDECODECREATEINFO get_decoder_info();

  const std::vector<char>& get_metadata_bytes();

  const std::vector<char>& get_bitstream_bytes();

  const std::vector<int64_t>& get_keyframe_positions();

  const std::vector<int64_t>& get_keyframe_timestamps();

  const std::vector<int64_t>& get_keyframe_byte_offsets();

 private:
  AVCodecContext* cc_;

  int prev_frame_;

  bool is_metadata_;
  bool is_keyframe_;

  std::vector<char> metadata_packets_;
  std::vector<char> bitstream_packets_;

  std::vector<int64_t> keyframe_positions_;
  std::vector<int64_t> keyframe_timestamps_;
  std::vector<int64_t> keyframe_byte_offsets_;

  double decode_time_;
};
}
