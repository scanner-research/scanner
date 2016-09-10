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

#include "scanner/util/queue.h"
#include "scanner/util/profiler.h"
#include "scanner/util/common.h"

#include <string>
#include <pthread.h>

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavformat/avio.h"
#include "libavutil/error.h"
#include "libswscale/swscale.h"
}

namespace scanner {

extern pthread_mutex_t av_mutex;

///////////////////////////////////////////////////////////////////////////////
/// VideoDecoder
class VideoDecoder {
public:
  VideoDecoder(CUcontext cuda_context, DatasetItemMetadata metadata);

  ~VideoDecoder();

  bool feed(
    const char* encoded_buffer,
    size_t encoded_size,
    bool discontinuity = false);

  bool discard_frame();

  bool get_frame(
    char* decoded_buffer,
    size_t decoded_size);

  int decoded_frames_buffered();

  void wait_until_frames_copied();

  void set_profiler(Profiler* profiler);

private:
  static int cuvid_handle_video_sequence(
    void *opaque,
    CUVIDEOFORMAT* format);

  static int cuvid_handle_picture_decode(
    void *opaque,
    CUVIDPICPARAMS* picparams);

  static int cuvid_handle_picture_display(
    void *opaque,
    CUVIDPARSERDISPINFO* dispinfo);

  const int max_output_frames_;
  const int max_mapped_frames_;

  std::vector<cudaStream_t> streams_;
  CUcontext cuda_context_;
  DatasetItemMetadata metadata_;
  std::vector<char> metadata_packets_;

  CUvideoparser parser_;
  CUvideodecoder decoder_;

  Queue<CUVIDPARSERDISPINFO> frame_queue_;
  std::vector<CUdeviceptr> mapped_frames_;

  int prev_frame_;

  int wait_for_iframe_;

  double decode_time_;

  Profiler* profiler_;
};



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
