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

struct VideoMetadata {
  int32_t frames;
  int32_t width;
  int32_t height;
};

class VideoDecoder {
public:
  VideoDecoder(VideoMetadata metadata);

#ifdef HARDWARE_DECODE
  VideoDecoder(CUcontext cuda_context, VideoMetadata metadata);
#endif

  ~VideoDecoder();

  AVFrame* decode(char* buffer, size_t size);

  double time_spent_on_decode();

private:
  VideoMetadata metadata_;

  AVPacket packet_;
  std::vector<AVFrame*> buffered_frames_;
  AVCodec* codec_;
  AVCodecContext* cc_;

  int next_frame_;
  int next_buffered_frame_;
  int buffered_frame_pos_;
  bool near_eof_;

  double decode_time_;
};

void preprocess_video(
  StorageBackend* storage,
  const std::string& video_path,
  const std::string& processed_video_path,
  const std::string& video_metadata_path,
  const std::string& iframe_path);

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
