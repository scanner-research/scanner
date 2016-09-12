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

#include "scanner/video/software/software_video_decoder.h"

namespace scanner {

///////////////////////////////////////////////////////////////////////////////
/// SoftwareVideoDecoder
SoftwareVideoDecoder::SoftwareVideoDecoder(
  DatasetItemMetadata metadata,
  int device_id)
{
}

SoftwareVideoDecoder::~SoftwareVideoDecoder() {
}

bool SoftwareVideoDecoder::feed(
  const char* encoded_buffer,
  size_t encoded_size,
  bool discontinuity)
{
  return false;
}

bool SoftwareVideoDecoder::discard_frame() {
  return false;
}

bool SoftwareVideoDecoder::get_frame(
  char* decoded_buffer,
  size_t decoded_size) 
{
  return false;
}

int SoftwareVideoDecoder::decoded_frames_buffered() {
  return 0;
}

void SoftwareVideoDecoder::wait_until_frames_copied() {
}

}
