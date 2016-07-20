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

#include "lightscan/storage/storage_backend.h"
#include <string>
#include <pthread.h>

#ifdef HARDWARE_DECODE
#include <cuda.h>
#endif

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavformat/avio.h"
#include "libavformat/movenc.h"
#include "libavutil/error.h"
#include "libswscale/swscale.h"
}

namespace lightscan {

extern pthread_mutex_t av_mutex;

class VideoDecoder {
public:
  VideoDecoder(
    RandomReadFile* file,
    const std::vector<int>& keyframe_positions,
    const std::vector<int64_t>& keyframe_timestamps);

#ifdef HARDWARE_DECODE
  VideoDecoder(
    CUcontext cuda_context,
    RandomReadFile* file,
    const std::vector<int>& keyframe_positions,
    const std::vector<int64_t>& keyframe_timestamps);
#endif

  ~VideoDecoder();

  void seek(int frame_position);

  AVFrame* decode();

  double time_spent_on_io();

  double time_spent_on_decode();

private:
  struct RandomReadFileData {
    RandomReadFile* file;
    uint64_t pos; // current position
    uint64_t total_size;
  } buffer_;

  static int read_packet_fn(void *opaque, uint8_t *buf, int buf_size);

  static int64_t seek_fn(void *opaque, int64_t offset, int whence);

  void setup_format_context();

  void setup_video_stream_codec();

  RandomReadFile* file_;
  std::vector<int> keyframe_positions_;
  std::vector<int64_t> keyframe_timestamps_;

  AVPacket packet_;
  std::vector<AVFrame*> buffered_frames_;
  AVFormatContext* format_context_;
  AVIOContext* io_context_;
  AVCodec* codec_;
  AVCodecContext* cc_;
  int video_stream_index_;

  int next_frame_;
  int next_buffered_frame_;
  int buffered_frame_pos_;
  bool near_eof_;

  double io_time_;
  double decode_time_;
};

void preprocess_video(
  StorageBackend* storage,
  const std::string& video_path,
  const std::string& processed_video_path,
  const std::string& video_metadata_path,
  const std::string& iframe_path);

struct VideoMetadata {
  int32_t frames;
  int32_t width;
  int32_t height;
};

uint64_t read_video_metadata(
  RandomReadFile* file,
  uint64_t file_pos,
  VideoMetadata& meta);

uint64_t read_keyframe_info(
  RandomReadFile* file,
  uint64_t file_pos,
  std::vector<int>& keyframe_postions,
  std::vector<int64_t>& keyframe_timestamps);

}
