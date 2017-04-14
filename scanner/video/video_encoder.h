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

#include "scanner/api/kernel.h"
#include "scanner/engine/metadata.h"
#include "scanner/util/common.h"
#include "scanner/util/profiler.h"

#include <vector>

namespace scanner {
namespace internal {

class InputFormat;

enum class VideoEncoderType {
  NVIDIA,
  INTEL,
  SOFTWARE,
};

///////////////////////////////////////////////////////////////////////////////
/// VideoEncoder
class VideoEncoder {
 public:
  static std::vector<VideoEncoderType> get_supported_encoder_types();

  static bool has_encoder_type(VideoEncoderType type);

  static VideoEncoder* make_from_config(DeviceHandle device_handle,
                                        i32 num_devices, VideoEncoderType type);

  virtual ~VideoEncoder(){};

  virtual void configure(const FrameInfo& metadata) = 0;

  virtual bool feed(const u8* frame_buffer, size_t frame_size) = 0;

  virtual bool flush() = 0;

  virtual bool get_packet(u8* decoded_buffer, size_t decoded_size,
                          size_t& actual_packet_size) = 0;

  virtual int decoded_packets_buffered() = 0;

  virtual void wait_until_packets_copied() = 0;

  void set_profiler(Profiler* profiler);

 protected:
  Profiler* profiler_ = nullptr;
};
}
}
