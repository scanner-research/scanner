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

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/frame.h"
#include "libavutil/error.h"
#include "libavutil/opt.h"
}

#include <cassert>

namespace scanner {

///////////////////////////////////////////////////////////////////////////////
/// SoftwareVideoDecoder
SoftwareVideoDecoder::SoftwareVideoDecoder(
  DatasetItemMetadata metadata,
  int device_id)
  : codec_(nullptr),
    cc_(nullptr)
{
  av_init_packet(&packet_);

  codec_ = avcodec_find_decoder(AV_CODEC_ID_H264);
  if (!codec_) {
    fprintf(stderr, "could not find h264 decoder\n");
    exit(EXIT_FAILURE);
  }

  cc_ = avcodec_alloc_context3(codec_);
  if (!cc_) {
    fprintf(stderr, "could not alloc codec context");
    exit(EXIT_FAILURE);
  }

  if (avcodec_open2(cc_, codec_, NULL) < 0) {
    fprintf(stderr, "could not open codec\n");
    assert(false);
  }
}

SoftwareVideoDecoder::~SoftwareVideoDecoder() {
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(55, 53, 0)
  avcodec_free_context(&cc_);
#else
  avcodec_close(cc_);
  av_freep(&cc_);
#endif
  for (AVFrame* frame : frame_pool_) {
    av_frame_free(&frame);
  }
  for (AVFrame* frame : decoded_frame_queue_) {
    av_frame_free(&frame);
  }
}

bool SoftwareVideoDecoder::feed(
  const char* encoded_buffer,
  size_t encoded_size,
  bool discontinuity)
{
  if (av_new_packet(&packet_, encoded_size) < 0) {
    fprintf(stderr, "could not allocated packet for feeding into decoder\n");
    assert(false);
  }
  memcpy(packet_.data, encoded_buffer, encoded_size);

  uint8_t* orig_data = packet_.data;
  int orig_size = packet_.size;
  while (packet_.size > 0) {
    // Get frame from pool of allocated frames to decode video into
    if (frame_pool_.empty()) {
      // Create a new frame if our pool is empty
      frame_pool_.push_back(av_frame_alloc());
    }
    AVFrame* frame = frame_pool_.back();
    frame_pool_.pop_back();

    int got_picture = 0;
    int consumed_length =
      avcodec_decode_video2(cc_, frame, &got_picture, &packet_);
    if (consumed_length < 0) {
      char err_msg[256];
      av_strerror(consumed_length, err_msg, 256);
      fprintf(stderr, "Error while decoding frame (%d): %s\n",
              consumed_length, err_msg);
      assert(false);
    }
    if (got_picture) {
      if (frame->buf[0] == NULL) {
        // Must copy packet as data is stored statically
        AVFrame* cloned_frame = av_frame_clone(frame);
        if (cloned_frame == NULL) {
          fprintf(stderr, "could not clone frame\n");
          assert(false);
        }
        decoded_frame_queue_.push_back(cloned_frame);
        av_frame_free(&frame);
      } else {
        // Frame is reference counted so we can just take it directly
        decoded_frame_queue_.push_back(frame);
      }
    }
    packet_.data += consumed_length;
    packet_.size -= consumed_length;
  }
  packet_.data = orig_data;
  packet_.size = orig_size;
  av_packet_unref(&packet_);

  return decoded_frame_queue_.size() > 0;
}

bool SoftwareVideoDecoder::discard_frame() {
  AVFrame* frame = decoded_frame_queue_.front();
  decoded_frame_queue_.pop_front();
  av_frame_unref(frame);
  frame_pool_.push_back(frame);

  return decoded_frame_queue_.size() > 0;
}

bool SoftwareVideoDecoder::get_frame(
  char* decoded_buffer,
  size_t decoded_size) 
{
  int64_t size_left = decoded_size;
  
  AVFrame* frame = decoded_frame_queue_.front();
  decoded_frame_queue_.pop_front();
  for (int i = 0; i < 3; ++i) {
    for (int line = 0; line < frame->height; ++line) {
      memcpy(decoded_buffer,
             frame->data[i] + frame->linesize[i] * line,
             std::min(size_left, static_cast<int64_t>(frame->width)));
      decoded_buffer += frame->width;
      size_left -= frame->width;
      if (size_left <= 0) {
        line = frame->height;
        i = 3;
      }
    }
  }
  av_frame_unref(frame);
  frame_pool_.push_back(frame);

  return decoded_frame_queue_.size() > 0;
}

int SoftwareVideoDecoder::decoded_frames_buffered() {
  return decoded_frame_queue_.size();
}

void SoftwareVideoDecoder::wait_until_frames_copied() {
}

}
