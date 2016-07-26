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
#include "lightscan/util/queue.h"
#include <string>
#include <pthread.h>

#include <cuda.h>
#include <nvcuvid.h>

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
  cudaVideoCodec codec_type;
  cudaVideoChromaFormat chroma_format;
};

class VideoSeparator {
public:
  VideoSeparator(CUcontext cuda_context, AVCodecContext* cc);

  ~VideoSeparator();

  bool decode(AVPacket* packet);

  const std::vector<char>& get_metadata_bytes();

  const std::vector<char>& get_bitstream_bytes();

  const std::vector<int>& get_keyframe_positions();

  const std::vector<int64_t>& get_keyframe_byte_offsets();

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

  CUcontext cuda_context_;
  AVCodecContext* cc_;

  CUvideoparser parser_;
  CUVIDDECODECREATEINFO decoder_info_;

  int prev_frame_;

  bool is_metadata_;
  bool is_keyframe_;

  std::vector<char> metadata_packets_;
  std::vector<char> bitstream_packets_;

  std::vector<int> keyframe_positions_;
  std::vector<int64_t> keyframe_byte_offsets_;

  double decode_time_;
};

class VideoDecoder {
public:
  VideoDecoder(CUcontext cuda_context,
               VideoMetadata metadata,
               std::vector<char> metadata_bytes);

  ~VideoDecoder();

  bool decode(
    const char* encoded_buffer,
    size_t encoded_size,
    char* decoded_buffer,
    size_t decoded_size);

  double time_spent_on_decode();

  void reset_timing();

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

  CUcontext cuda_context_;
  VideoMetadata metadata_;

  CUvideoparser parser_;
  CUvideodecoder decoder_;

  Queue<CUVIDPARSERDISPINFO> frame_queue_;

  int prev_frame_;

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
  VideoMetadata& meta,
  std::vector<char>& metadata_packets);

uint64_t read_keyframe_info(
  RandomReadFile* file,
  uint64_t file_pos,
  std::vector<int>& keyframe_postions,
  std::vector<int64_t>& keyframe_timestamps);

}
