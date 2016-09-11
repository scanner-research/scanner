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

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvcuvid.h>

namespace scanner {

///////////////////////////////////////////////////////////////////////////////
/// NVIDIAVideoDecoder
class NVIDIAVideoDecoder : public VideoDecoder {
public:
  NVIDIAVideoDecoder(
    DatasetItemMetadata metadata,
    int device_id,
    CUcontext cuda_context);

  ~NVIDIAVideoDecoder();

  bool feed(
    const char* encoded_buffer,
    size_t encoded_size,
    bool discontinuity = false) override;

  bool discard_frame() override;

  bool get_frame(
    char* decoded_buffer,
    size_t decoded_size) override;

  int decoded_frames_buffered() override;

  void wait_until_frames_copied() override;

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

  DatasetItemMetadata metadata_;
  int device_id_;
  CUcontext cuda_context_;
  std::vector<char> metadata_packets_;
  const int max_output_frames_;
  const int max_mapped_frames_;
  std::vector<cudaStream_t> streams_;
  CUvideoparser parser_;
  CUvideodecoder decoder_;
  Queue<CUVIDPARSERDISPINFO> frame_queue_;
  std::vector<CUdeviceptr> mapped_frames_;
  int prev_frame_;
  int wait_for_iframe_;
};

}
