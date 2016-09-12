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

#include "scanner/video/video_decoder.h"

namespace scanner {

///////////////////////////////////////////////////////////////////////////////
/// SoftwareVideoDecoder
class SoftwareVideoDecoder : public VideoDecoder {
public:
  SoftwareVideoDecoder(
    DatasetItemMetadata metadata,
    int device_id);

  ~SoftwareVideoDecoder();

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
  DatasetItemMetadata metadata_;
  int device_id_;
  int prev_frame_;
  int wait_for_iframe_;
};

}
