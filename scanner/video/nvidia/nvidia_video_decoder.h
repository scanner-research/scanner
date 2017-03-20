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

#include "scanner/api/kernel.h"
#include "scanner/util/common.h"
#include "scanner/util/queue.h"
#include "scanner/video/video_decoder.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvcuvid.h>

namespace scanner {
namespace internal {

///////////////////////////////////////////////////////////////////////////////
/// NVIDIAVideoDecoder
class NVIDIAVideoDecoder : public VideoDecoder {
 public:
  NVIDIAVideoDecoder(int device_id, DeviceType output_type,
                     CUcontext cuda_context);

  ~NVIDIAVideoDecoder();

  void configure(const FrameInfo& metadata) override;

  bool feed(const u8* encoded_buffer, size_t encoded_size,
            bool discontinuity = false) override;

  bool discard_frame() override;

  bool get_frame(u8* decoded_buffer, size_t decoded_size) override;

  int decoded_frames_buffered() override;

  void wait_until_frames_copied() override;

 private:
  static int cuvid_handle_video_sequence(void* opaque, CUVIDEOFORMAT* format);

  static int cuvid_handle_picture_decode(void* opaque,
                                         CUVIDPICPARAMS* picparams);

  static int cuvid_handle_picture_display(void* opaque,
                                          CUVIDPARSERDISPINFO* dispinfo);

  int device_id_;
  DeviceType output_type_;
  CUcontext cuda_context_;
  static const int max_output_frames_ = 32;
  static const int max_mapped_frames_ = 8;
  std::vector<cudaStream_t> streams_;

  i32 frame_width_;
  i32 frame_height_;
  std::vector<char> metadata_packets_;
  CUvideoparser parser_;
  CUvideodecoder decoder_;

  i32 last_displayed_frame_;
  volatile i32 frame_in_use_[max_output_frames_];
  volatile i32 undisplayed_frames_[max_output_frames_];
  volatile i32 invalid_frames_[max_output_frames_];

  std::mutex frame_queue_mutex_;
  CUVIDPARSERDISPINFO frame_queue_[max_output_frames_];
  i32 frame_queue_read_pos_;
  i32 frame_queue_elements_;

  CUdeviceptr mapped_frames_[max_mapped_frames_];
};
}
}
