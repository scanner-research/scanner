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

#include "scanner/video/software/software_video_encoder.h"
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

#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 5, 0)
#define PACKET_FREE(pkt) \
  av_packet_free(&pkt);
#else
#define PACKET_FREE(pkt) \
  av_packet_unref(pkt);  \
  av_freep(&pkt);
#endif

namespace scanner {
namespace internal {

///////////////////////////////////////////////////////////////////////////////
/// SoftwareVideoEncoder
SoftwareVideoEncoder::SoftwareVideoEncoder(i32 device_id,
                                           DeviceType output_type,
                                           i32 thread_count)
    : device_id_(device_id),
      output_type_(output_type),
      codec_(nullptr),
      cc_(nullptr),
      sws_context_(nullptr),
      was_reset_(false),
      ready_packet_queue_(1024),
      frame_(nullptr) {
  avcodec_register_all();

  codec_ = avcodec_find_encoder(AV_CODEC_ID_H264);
  if (!codec_) {
    fprintf(stderr, "could not find h264 encoder\n");
    exit(EXIT_FAILURE);
  }

  annexb_ = av_bitstream_filter_init("h264_mp4toannexb");
}

SoftwareVideoEncoder::~SoftwareVideoEncoder() {
  if (cc_) {
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(55, 53, 0)
    avcodec_free_context(&cc_);
#else
    avcodec_close(cc_);
    av_freep(&cc_);
#endif
  }
  if (frame_) {
    av_frame_free(&frame_);
  }

  if (sws_context_) {
    sws_freeContext(sws_context_);
  }

  av_bitstream_filter_close(annexb_);
}

void SoftwareVideoEncoder::configure(const FrameInfo& metadata) {
  if (cc_ != NULL) {
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(55, 53, 0)
    avcodec_free_context(&cc_);
#else
    avcodec_close(cc_);
    av_freep(&cc_);
#endif
    while (ready_packet_queue_.size() > 0) {
      AVPacket* packet;
      ready_packet_queue_.pop(packet);
      PACKET_FREE(packet);
    }
  }

  cc_ = avcodec_alloc_context3(codec_);
  if (!cc_) {
    fprintf(stderr, "could not alloc codec context");
    exit(EXIT_FAILURE);
  }

  metadata_ = metadata;
  frame_width_ = metadata_.width();
  frame_height_ = metadata_.height();

  int required_size = av_image_get_buffer_size(AV_PIX_FMT_RGB24, frame_width_,
                                               frame_height_, 1);

  cc_->thread_count = 4;
  cc_->bit_rate = 8 * 1024 * 1024;  // Bits Per Second
  cc_->width = frame_width_;     // Note Resolution must be a multiple of 2!!
  cc_->height = frame_height_;   // Note Resolution must be a multiple of 2!!
  // TODO(apoms): figure out this fps from the input video automatically
  cc_->time_base.den = 24;
  cc_->time_base.num = 1;
  cc_->gop_size = 120;  // Intra frames per x P frames
  cc_->pix_fmt =
    AV_PIX_FMT_YUV420P;  // Do not change this, H264 needs YUV format not RGB

  if (avcodec_open2(cc_, codec_, NULL) < 0) {
    fprintf(stderr, "could not open codec\n");
    assert(false);
  }

  AVPixelFormat encoder_pixel_format = cc_->pix_fmt;
  sws_context_ = sws_getContext(
    frame_width_, frame_height_, AV_PIX_FMT_RGB24,
    frame_width_, frame_height_, encoder_pixel_format,
    SWS_BICUBIC, NULL, NULL, NULL);
  if (sws_context_ == NULL) {
    fprintf(stderr, "Could not get sws_context for rgb conversion\n");
    exit(EXIT_FAILURE);
  }

}

bool SoftwareVideoEncoder::feed(const u8* frame_buffer, size_t frame_size) {
  assert(frame_size > 0);
  if (was_reset_) {
    avcodec_flush_buffers(cc_);
  }

  // Convert image into YUV format from RGB
  frame_ = av_frame_alloc();
  if (!frame_) {
    fprintf(stderr, "Could not alloc frame\n");
    exit(EXIT_FAILURE);
  }

  frame_->format = cc_->pix_fmt;
  frame_->width = frame_width_;
  frame_->height = frame_height_;

  if (av_frame_get_buffer(frame_, 32) < 0) {
    fprintf(stderr, "Could not get frame buffer\n");
    exit(EXIT_FAILURE);
  }

  uint8_t* out_slices[4];
  int out_linesizes[4];
  int required_size =
    av_image_fill_arrays(out_slices, out_linesizes, frame_buffer,
                         AV_PIX_FMT_RGB24, frame_width_, frame_height_, 1);
  if (required_size < 0) {
    fprintf(stderr, "Error in av_image_fill_arrays\n");
    exit(EXIT_FAILURE);
  }
  if (required_size > frame_size) {
    fprintf(stderr, "Encode buffer not large enough for image\n");
    exit(EXIT_FAILURE);
  }
  auto scale_start = now();
  if (sws_scale(sws_context_, out_slices, out_linesizes, 0, frame_height_,
                frame_->data, frame_->linesize) < 0) {
    fprintf(stderr, "sws_scale failed\n");
    exit(EXIT_FAILURE);
  }
  auto scale_end = now();
  if (profiler_) {
    profiler_->add_interval("ffmpeg:scale_frame", scale_start, scale_end);
  }

  feed_frame(false);

  return ready_packet_queue_.size() > 0;
}

bool SoftwareVideoEncoder::flush() {
  feed_frame(false);
  was_reset_ = true;
  return ready_packet_queue_.size() > 0;
}

bool SoftwareVideoEncoder::get_packet(u8* packet_buffer, size_t packet_size,
                                      size_t& actual_packet_size) {
  actual_packet_size = 0;

  AVPacket* packet;
  if (ready_packet_queue_.size() > 0) {
    ready_packet_queue_.peek(packet);
  } else {
    return false;
  }

  u8* filtered_data;
  i32 filtered_data_size;
  int err = av_bitstream_filter_filter(
    annexb_, cc_, NULL, &filtered_data, &filtered_data_size, packet->data,
    packet->size, packet->flags & AV_PKT_FLAG_KEY);
  if (err < 0) {
    char err_msg[256];
    av_strerror(err, err_msg, 256);
    LOG(ERROR) << "Error while filtering: " << err_msg;
    exit(1);
  }

  // Make sure we have space for this packet, otherwise return
  actual_packet_size = filtered_data_size;
  if (actual_packet_size > packet_size) {
    return true;
  }

  memcpy(packet_buffer, filtered_data, filtered_data_size);
  free(filtered_data);

  // Only pop packet when we know we can copy it out
  ready_packet_queue_.pop(packet);
  PACKET_FREE(packet);

  return ready_packet_queue_.size() > 0;
}

int SoftwareVideoEncoder::decoded_packets_buffered() {
  return ready_packet_queue_.size();
}

void SoftwareVideoEncoder::wait_until_packets_copied() {}

void SoftwareVideoEncoder::feed_frame(bool flush) {
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 25, 0)
  auto send_start = now();
  AVFrame* f = flush ? NULL : frame_;
  int ret = avcodec_send_frame(cc_, f);
  if (ret != AVERROR_EOF) {
    if (ret < 0) {
      char err_msg[256];
      av_strerror(ret, err_msg, 256);
      fprintf(stderr, "Error while sending frame (%d): %s\n", ret, err_msg);
      exit(1);
    }
  }

  auto send_end = now();

  auto receive_start = now();
  while (ret == 0) {
    AVPacket* packet = av_packet_alloc();
    ret = avcodec_receive_packet(cc_, packet);
    if (ret == 0) {
      ready_packet_queue_.push(packet);
    } else if (ret == AVERROR(EAGAIN)) {
      PACKET_FREE(packet);
    } else if (ret == AVERROR_EOF) {
      PACKET_FREE(packet);
    } else {
      char err_msg[256];
      av_strerror(ret, err_msg, 256);
      fprintf(stderr, "Error while receiving packet (%d): %s\n", ret,
              err_msg);
      exit(1);
    }
  }
  auto receive_end = now();

  if (f) {
    av_frame_free(&frame_);
    frame_ = nullptr;
  }
#else
  auto send_start = now();
  auto send_end = now();
  auto receive_start = now();
  auto receive_end = now();
  LOG(FATAL) << "Not supported";
#endif
  if (profiler_) {
    profiler_->add_interval("ffmpeg:send_frame", send_start, send_end);
    profiler_->add_interval("ffmpeg:receive_packet", receive_start,
                            receive_end);
  }
}

}
}
