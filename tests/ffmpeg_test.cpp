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

#include "scanner/engine/db.h"
#include "scanner/util/fs.h"
#include "scanner/util/h264.h"
#include "scanner/util/queue.h"
#include "tests/videos.h"
#include "storehouse/storage_backend.h"

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

#include <gtest/gtest.h>

#include <cassert>

using namespace scanner::internal;

namespace scanner {

// Fixtures are taken down after every test, so to avoid-redownloading and
// ingesting the files, we use static globals.
static bool downloaded = false;
static std::string video_path;

struct DecoderState {
  AVPacket packet;
  AVCodec* codec;
  AVCodecContext* cc;
  SwsContext* sws_context;

  //AVFrame* frame;
  Queue<AVFrame*> pool;
  Queue<AVFrame*> frame;

  i32 frame_width;
  i32 frame_height;

  DecoderState()
      :pool(100000), frame(100000) {
    av_init_packet(&packet);

    codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    EXPECT_TRUE(codec) << "could not find h264 decoder";

    cc = avcodec_alloc_context3(codec);
    EXPECT_TRUE(cc) << "could not alloc codec context";

    cc->thread_count = 16;

    int result = avcodec_open2(cc, codec, NULL);
    EXPECT_TRUE(result >= 0) << "could not open codec";

    sws_context = nullptr;

  };

  ~DecoderState() {
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(55, 53, 0)
    avcodec_free_context(&cc);
#else
    avcodec_close(cc);
    av_freep(&cc);
#endif
    while (frame.size() > 0) {
      AVFrame* f;
      frame.pop(f);
      av_frame_free(&f);
    }
    sws_freeContext(sws_context);
  }

  void feed_frame(const u8 *encoded_buffer, i32 encoded_size) {
    if (encoded_size > 0) {
      if (av_new_packet(&packet, encoded_size) < 0) {
        fprintf(stderr, "could not allocate packet for feeding into decoder\n");
        assert(false);
      }
      memcpy(packet.data, encoded_buffer, encoded_size);
    } else {
      packet.data = NULL;
      packet.size = 0;
    }
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 25, 0)
    int error = avcodec_send_packet(cc, &packet);
    if (error != AVERROR_EOF) {
      if (error < 0) {
        char err_msg[256];
        av_strerror(error, err_msg, 256);
        fprintf(stderr, "Error while sending packet (%d): %s\n", error,
                err_msg);
        exit(1);
      }
    }

    bool done = false;
    while (!done) {
      AVFrame* f;
      if (pool.size() <= 0) {
        pool.push(av_frame_alloc());
      }
      pool.pop(f);

      error = avcodec_receive_frame(cc, f);
      if (error == AVERROR_EOF) {
        pool.push(f);
        break;
      }
      if (error == 0) {
        frame.push(f);
        continue;
      } else if (error == AVERROR(EAGAIN)) {
        pool.push(f);
        done = true;
      } else {
        char err_msg[256];
        av_strerror(error, err_msg, 256);
        fprintf(stderr, "Error while receiving frame (%d): %s\n", error,
                err_msg);
        assert(false);
      }
    }
#else
  // uint8_t *orig_data = packet_.data;
  // int orig_size = packet_.size;
  // int got_picture = 0;
  // do {
  //   // Get frame from pool of allocated frames to decode video into
  //   AVFrame *frame;
  //   {
  //     std::lock_guard<std::mutex> lock(frame_mutex_);
  //     if (frame_pool_.empty()) {
  //       // Create a new frame if our pool is empty
  //       frame_pool_.push_back(av_frame_alloc());
  //     }
  //     frame = frame_pool_.back();
  //     frame_pool_.pop_back();
  //   }

  //   auto decode_start = now();
  //   int consumed_length =
  //       avcodec_decode_video2(cc_, frame, &got_picture, &packet_);
  //   if (profiler_) {
  //     profiler_->add_interval("ffmpeg:decode_video", decode_start, now());
  //   }
  //   if (consumed_length < 0) {
  //     char err_msg[256];
  //     av_strerror(consumed_length, err_msg, 256);
  //     fprintf(stderr, "Error while decoding frame (%d): %s\n", consumed_length,
  //             err_msg);
  //     assert(false);
  //   }
  //   if (got_picture) {
  //     if (frame->buf[0] == NULL) {
  //       // Must copy packet as data is stored statically
  //       AVFrame *cloned_frame = av_frame_clone(frame);
  //       if (cloned_frame == NULL) {
  //         fprintf(stderr, "could not clone frame\n");
  //         assert(false);
  //       }
  //       std::lock_guard<std::mutex> lock(frame_mutex_);
  //       printf("clone\n");
  //       decoded_frame_queue_.push_back(cloned_frame);
  //       av_frame_unref(frame);
  //       frame_pool_.push_back(frame);
  //     } else {
  //       // Frame is reference counted so we can just take it directly
  //       std::lock_guard<std::mutex> lock(frame_mutex_);
  //       printf("push\n");
  //       decoded_frame_queue_.push_back(frame);
  //     }
  //   } else {
  //     std::lock_guard<std::mutex> lock(frame_mutex_);
  //     frame_pool_.push_back(frame);
  //   }
  //   packet_.data += consumed_length;
  //   packet_.size -= consumed_length;
  // } while (packet_.size > 0 || (orig_size == 0 && got_picture));
  // packet_.data = orig_data;
  // packet_.size = orig_size;
#endif
  av_packet_unref(&packet);
}

void get_frame(u8* decoded_buffer, i32 decoded_size) {
  if (frame.size() <= 0) {
    return;
  }

  AVFrame* f;
  frame.pop(f);

  printf("decode\n");
  if (sws_context == nullptr) {
    AVPixelFormat decoder_pixel_format = cc->pix_fmt;
    sws_context = sws_getContext(
        frame_width, frame_height, decoder_pixel_format, frame_width,
        frame_height, AV_PIX_FMT_RGB24, SWS_BICUBIC, NULL, NULL, NULL);
  }

  if (sws_context == NULL) {
    fprintf(stderr, "Could not get sws_context for rgb conversion\n");
    exit(EXIT_FAILURE);
  }

  u8 *scale_buffer = decoded_buffer;

  uint8_t *out_slices[4];
  int out_linesizes[4];
  int required_size = av_image_fill_arrays(out_slices, out_linesizes,
                                           scale_buffer, AV_PIX_FMT_RGB24,
                                           frame_width, frame_height, 1);
  if (required_size < 0) {
    fprintf(stderr, "Error in av_image_fill_arrays\n");
    exit(EXIT_FAILURE);
  }
  if (required_size > decoded_size) {
    fprintf(stderr, "Decode buffer not large enough for image\n");
    exit(EXIT_FAILURE);
  }
  if (sws_scale(sws_context, f->data, f->linesize, 0,
                f->height, out_slices, out_linesizes) < 0) {
    fprintf(stderr, "sws_scale failed\n");
    exit(EXIT_FAILURE);
  }
  sws_freeContext(sws_context);
  sws_context = nullptr;

  pool.push(f);
}

};

class FfmpegTest : public ::testing::Test {
protected:
  void SetUp() override {
    avcodec_register_all();

    // Ingest video
    if (!downloaded) {
      downloaded = true;
      video_path = download_video(short_video);
    }
  }

  void TearDown() {
  }

  void gen_random(char *s, const int len) {
    static const char alphanum[] =
      "0123456789"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz";

    for (int i = 0; i < len; ++i) {
      s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
    }

    s[len] = 0;
  }
};


TEST_F(FfmpegTest, MemoryLeak) {
  std::unique_ptr<storehouse::StorageConfig> sc(
      storehouse::StorageConfig::make_posix_config());
  auto storage = storehouse::StorageBackend::make_from_config(sc.get());

  // Load test data
  printf("Reading data...\n");
  VideoMetadata video_meta =
      read_video_metadata(storage, download_video_meta(short_video));
  std::vector<u8> video_bytes = read_entire_file(video_path);

  DecoderState ds;
  ds.frame_width = video_meta.width();
  ds.frame_height = video_meta.height();
  int required_size = av_image_get_buffer_size(AV_PIX_FMT_RGB24, ds.frame_width,
                                               ds.frame_height, 1);

  u8* decode_buffer = new u8[required_size];

  const u8 *encoded_buffer = (const u8 *)video_bytes.data();
  size_t encoded_buffer_size = video_bytes.size();

  printf("Starting decoding...\n");
  int iterations = 3;
  for (int i = 0; i < iterations; ++i) {
    printf("Iteration %d\n", i);
    size_t buffer_offset = 0;
    while (buffer_offset < encoded_buffer_size) {
      i32 encoded_packet_size = 0;
      const u8 *encoded_packet = NULL;
      if (buffer_offset < encoded_buffer_size) {
        encoded_packet_size = *reinterpret_cast<const i32 *>(
            encoded_buffer + buffer_offset);
        buffer_offset += sizeof(i32);
        encoded_packet = encoded_buffer + buffer_offset;
        assert(encoded_packet_size < encoded_buffer_size);
        buffer_offset += encoded_packet_size;
      }

      ds.feed_frame(encoded_packet, encoded_packet_size);
      if (buffer_offset < encoded_buffer_size) {
        ds.get_frame(decode_buffer, required_size);
      }
    }
  }
  delete[] decode_buffer;
}

}
