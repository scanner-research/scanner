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
#include "scanner/util/h264.h"

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/error.h"
#include "libavutil/frame.h"
#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
#include "libswscale/swscale.h"
}

#ifdef HAVE_CUDA
#include "scanner/util/cuda.h"
#endif

#include <cassert>

namespace scanner {
namespace internal {

///////////////////////////////////////////////////////////////////////////////
/// SoftwareVideoDecoder
SoftwareVideoDecoder::SoftwareVideoDecoder(i32 device_id,
                                           DeviceType output_type,
                                           i32 thread_count)
    : device_id_(device_id),
      output_type_(output_type),
      codec_(nullptr),
      cc_(nullptr),
      reset_context_(true),
      sws_context_(nullptr),
      frame_pool_(1024),
      decoded_frame_queue_(1024) {
  avcodec_register_all();

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

  //cc_->thread_count = thread_count;
  cc_->thread_count = 4;

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
  while (frame_pool_.size() > 0) {
    AVFrame* frame;
    frame_pool_.pop(frame);
    av_frame_free(&frame);
  }
  while (decoded_frame_queue_.size() > 0) {
    AVFrame* frame;
    decoded_frame_queue_.pop(frame);
    av_frame_free(&frame);
  }

  sws_freeContext(sws_context_);
}

void SoftwareVideoDecoder::configure(const FrameInfo& metadata) {
  metadata_ = metadata;
  frame_width_ = metadata_.shape[1];
  frame_height_ = metadata_.shape[2];
  reset_context_ = true;

  int required_size = av_image_get_buffer_size(AV_PIX_FMT_RGB24, frame_width_,
                                               frame_height_, 1);

  conversion_buffer_.resize(required_size);
}

bool SoftwareVideoDecoder::feed(const u8* encoded_buffer, size_t encoded_size,
                                bool discontinuity) {
// Debug read packets
#if 0
  i32 es = (i32)encoded_size;
  const u8* b = encoded_buffer;
  while (es > 0) {
    const u8* nal_start;
    i32 nal_size;
    next_nal(b, es, nal_start, nal_size);
    i32 nal_unit_type = get_nal_unit_type(nal_start);
    printf("nal unit type %d\n", nal_unit_type);

    if (nal_unit_type == 7) {
      i32 offset = 32;
      i32 sps_id = parse_exp_golomb(nal_start, nal_size, offset);
      printf("SPS NAL (%d)\n", sps_id);
    }
    if (nal_unit_type == 8) {
      i32 offset = 8;
      i32 pps_id = parse_exp_golomb(nal_start, nal_size, offset);
      i32 sps_id = parse_exp_golomb(nal_start, nal_size, offset);
      printf("PPS id: %d, SPS id: %d\n", pps_id, sps_id);
    }
  }
#endif
  if (discontinuity) {
    while (decoded_frame_queue_.size() > 0) {
      AVFrame* frame;
      decoded_frame_queue_.pop(frame);
      av_frame_free(&frame);
    }
    while (frame_pool_.size() > 0) {
      AVFrame* frame;
      frame_pool_.pop(frame);
      av_frame_free(&frame);
    }

    packet_.data = NULL;
    packet_.size = 0;
    feed_packet(true);
    return false;
  }
  if (encoded_size > 0) {
    if (av_new_packet(&packet_, encoded_size) < 0) {
      fprintf(stderr, "could not allocate packet for feeding into decoder\n");
      assert(false);
    }
    memcpy(packet_.data, encoded_buffer, encoded_size);
  } else {
    packet_.data = NULL;
    packet_.size = 0;
  }

  feed_packet(false);
  av_packet_unref(&packet_);

  return decoded_frame_queue_.size() > 0;
}

bool SoftwareVideoDecoder::discard_frame() {
  if (decoded_frame_queue_.size() > 0) {
    AVFrame* frame;
    decoded_frame_queue_.pop(frame);
    av_frame_unref(frame);
    frame_pool_.push(frame);
  }

  return decoded_frame_queue_.size() > 0;
}

bool SoftwareVideoDecoder::get_frame(u8* decoded_buffer, size_t decoded_size) {
  int64_t size_left = decoded_size;

  AVFrame* frame;
  if (decoded_frame_queue_.size() > 0) {
    decoded_frame_queue_.pop(frame);
  } else {
    return false;
  }

  if (reset_context_) {
    auto get_context_start = now();
    AVPixelFormat decoder_pixel_format = cc_->pix_fmt;
    sws_freeContext(sws_context_);
    sws_context_ = sws_getContext(
        frame_width_, frame_height_, decoder_pixel_format, frame_width_,
        frame_height_, AV_PIX_FMT_RGB24, SWS_BICUBIC, NULL, NULL, NULL);
    reset_context_ = false;
    auto get_context_end = now();
    if (profiler_) {
      profiler_->add_interval("ffmpeg:get_sws_context", get_context_start,
                              get_context_end);
    }
  }

  if (sws_context_ == NULL) {
    fprintf(stderr, "Could not get sws_context for rgb conversion\n");
    exit(EXIT_FAILURE);
  }

  u8* scale_buffer = decoded_buffer;

  uint8_t* out_slices[4];
  int out_linesizes[4];
  int required_size =
      av_image_fill_arrays(out_slices, out_linesizes, scale_buffer,
                           AV_PIX_FMT_RGB24, frame_width_, frame_height_, 1);
  if (required_size < 0) {
    fprintf(stderr, "Error in av_image_fill_arrays\n");
    exit(EXIT_FAILURE);
  }
  if (required_size > decoded_size) {
    fprintf(stderr, "Decode buffer not large enough for image\n");
    exit(EXIT_FAILURE);
  }
  auto scale_start = now();
  if (sws_scale(sws_context_, frame->data, frame->linesize, 0, frame->height,
                out_slices, out_linesizes) < 0) {
    fprintf(stderr, "sws_scale failed\n");
    exit(EXIT_FAILURE);
  }
  auto scale_end = now();

  av_frame_unref(frame);
  frame_pool_.push(frame);

  if (profiler_) {
    profiler_->add_interval("ffmpeg:scale_frame", scale_start, scale_end);
  }

  return decoded_frame_queue_.size() > 0;
}

int SoftwareVideoDecoder::decoded_frames_buffered() {
  return decoded_frame_queue_.size();
}

void SoftwareVideoDecoder::wait_until_frames_copied() {}

void SoftwareVideoDecoder::feed_packet(bool flush) {
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 25, 0)
  auto send_start = now();
  int error = avcodec_send_packet(cc_, &packet_);
  if (error != AVERROR_EOF) {
    if (error < 0) {
      char err_msg[256];
      av_strerror(error, err_msg, 256);
      fprintf(stderr, "Error while sending packet (%d): %s\n", error, err_msg);
      assert(false);
    }
  }
  auto send_end = now();

  auto received_start = now();
  bool done = false;
  while (!done) {
    AVFrame* frame;
    {
      if (frame_pool_.size() <= 0) {
        // Create a new frame if our pool is empty
        frame_pool_.push(av_frame_alloc());
      }
      frame_pool_.pop(frame);
    }

    error = avcodec_receive_frame(cc_, frame);
    if (error == AVERROR_EOF) {
      frame_pool_.push(frame);
      break;
    }
    if (error == 0) {
      if (!flush) {
        decoded_frame_queue_.push(frame);
      } else {
        av_frame_unref(frame);
        frame_pool_.push(frame);
      }
    } else if (error == AVERROR(EAGAIN)) {
      done = true;
      frame_pool_.push(frame);
    } else {
      char err_msg[256];
      av_strerror(error, err_msg, 256);
      fprintf(stderr, "Error while receiving frame (%d): %s\n", error, err_msg);
      exit(1);
    }
  }
  auto received_end = now();
  if (profiler_) {
    profiler_->add_interval("ffmpeg:send_packet", send_start, send_end);
    profiler_->add_interval("ffmpeg:receive_frame", received_start,
                            received_end);
  }
#else
  uint8_t* orig_data = packet_.data;
  int orig_size = packet_.size;
  int got_picture = 0;
  do {
    // Get frame from pool of allocated frames to decode video into
    AVFrame* frame;
    {
      if (frame_pool_.size() <= 0) {
        // Create a new frame if our pool is empty
        frame_pool_.push(av_frame_alloc());
      }
      frame_pool_.pop(frame);
    }

    auto decode_start = now();
    int consumed_length =
      avcodec_decode_video2(cc_, frame, &got_picture, &packet_);
    if (profiler_) {
      profiler_->add_interval("ffmpeg:decode_video", decode_start, now());
    }
    if (consumed_length < 0) {
      char err_msg[256];
      av_strerror(consumed_length, err_msg, 256);
      fprintf(stderr, "Error while decoding frame (%d): %s\n", consumed_length,
              err_msg);
      assert(false);
    }
    if (got_picture) {
      if (!flush) {
        if (frame->buf[0] == NULL) {
          // Must copy packet as data is stored statically
          AVFrame* cloned_frame = av_frame_clone(frame);
          if (cloned_frame == NULL) {
            fprintf(stderr, "could not clone frame\n");
            assert(false);
          }
          decoded_frame_queue_.push(cloned_frame);
          av_frame_free(&frame);
        } else {
          // Frame is reference counted so we can just take it directly
          decoded_frame_queue_.push(frame);
        }
      } else {
        av_frame_unref(frame);
        frame_pool_.push(frame);
      }
    } else {
      frame_pool_.push(frame);
    }
    packet_.data += consumed_length;
    packet_.size -= consumed_length;
  } while (packet_.size > 0 || (orig_size == 0 && got_picture));
  packet_.data = orig_data;
  packet_.size = orig_size;
#endif
  if (packet_.size == 0) {
    avcodec_flush_buffers(cc_);
  }
}

}
}
