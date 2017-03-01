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

#include "scanner/video/decoder_automata.h"
#include "scanner/metadata.pb.h"

#include "scanner/util/h264.h"
#include "scanner/util/memory.h"

#include <thread>

namespace scanner {
namespace internal {

DecoderAutomata::DecoderAutomata(DeviceHandle device_handle, i32 num_devices,
                                 VideoDecoderType decoder_type)
    : device_handle_(device_handle), num_devices_(num_devices),
      decoder_type_(decoder_type),
      decoder_(VideoDecoder::make_from_config(device_handle, num_devices,
                                              decoder_type)),
      feeder_waiting_(false), not_done_(true), frames_retrieved_(0) {
  feeder_thread_ = std::thread(&DecoderAutomata::feeder, this);
}

DecoderAutomata::~DecoderAutomata() {
  {
    reset_current_frame_ = -1;
    frames_to_get_ = 0;
    frames_retrieved_ = 0;
    while (decoder_->discard_frame()) {
    }

    std::unique_lock<std::mutex> lk(feeder_mutex_);
    wake_feeder_.wait(lk, [this] { return feeder_waiting_.load(); });

    if (frames_retrieved_ > 0) {
      decoder_->feed(nullptr, 0, true);
    }
    not_done_ = false;
    feeder_waiting_ = false;
  }
  wake_feeder_.notify_one();
  feeder_thread_.join();
}

void DecoderAutomata::initialize(
    const std::vector<proto::DecodeArgs> &encoded_data) {
  assert(!encoded_data.empty());
  encoded_data_ = encoded_data;
  frame_size_ = encoded_data[0].width() * encoded_data[0].height() * 3;
  current_frame_ = encoded_data[0].start_keyframe();
  next_frame_.store(encoded_data[0].valid_frames(0), std::memory_order_release);
  retriever_data_idx_.store(0, std::memory_order_release);
  retriever_valid_idx_ = 0;
  reset_current_frame_ = -1;

  FrameInfo info;
  info.set_width(encoded_data[0].width());
  info.set_height(encoded_data[0].height());

  while (decoder_->discard_frame()) {
  }

  std::unique_lock<std::mutex> lk(feeder_mutex_);
  wake_feeder_.wait(lk, [this] { return feeder_waiting_.load(); });

  if (info_.width() != info.width()
      || info_.height() != info.height()) {
    decoder_->configure(info);
  }
  if (frames_retrieved_ > 0) {
    decoder_->feed(nullptr, 0, true);
  }

  feeder_data_idx_.store(0, std::memory_order_release);
  feeder_buffer_offset_.store(0, std::memory_order_release);
  feeder_next_keyframe_.store(encoded_data[0].keyframes(1),
                              std::memory_order_release);
  feeder_data_idx_.store(0);
  feeder_buffer_offset_.store(0);
  feeder_next_keyframe_.store(encoded_data[0].keyframes(1));
  info_ = info;
  std::atomic_thread_fence(std::memory_order_release);
  seeking_ = false;
}

void DecoderAutomata::get_frames(u8* buffer, i32 num_frames) {
  i64 total_frames_decoded = 0;
  i64 total_frames_used = 0;

  // profiler_->add_interval("decode", start, now());
  // profiler_->increment("effective_frames", total_frames_used);
  // profiler_->increment("decoded_frames", total_frames_decoded);

  auto start = now();

  // Wait until feeder is waiting
  {
    // Wait until frames are being requested
    std::unique_lock<std::mutex> lk(feeder_mutex_);
    wake_feeder_.wait(lk, [this] { return feeder_waiting_.load(); });
  }

  // Start up feeder thread
  {
    std::unique_lock<std::mutex> lk(feeder_mutex_);
    frames_retrieved_ = 0;
    frames_to_get_ = num_frames;
    feeder_waiting_ = false;
  }
  wake_feeder_.notify_one();

  while (frames_retrieved_ < frames_to_get_) {
    if (reset_current_frame_ != -1) {
      printf("reset current frame\n");
      current_frame_ = reset_current_frame_;
      reset_current_frame_ = -1;
    }
    if (decoder_->decoded_frames_buffered() > 0) {
      // New frames
      bool more_frames = true;
      while (more_frames && frames_retrieved_ < frames_to_get_) {
        const auto &valid_frames =
            encoded_data_[retriever_data_idx_].valid_frames();
        assert(current_frame_ <= valid_frames.Get(retriever_valid_idx_));
        // printf("has buffered frames, curr %d, next %d\n",
        //        current_frame_, valid_frames.Get(retriever_valid_idx_));
        if (current_frame_ == valid_frames.Get(retriever_valid_idx_)) {
          u8 *decoded_buffer = buffer + frames_retrieved_ * frame_size_;
          more_frames = decoder_->get_frame(decoded_buffer, frame_size_);
          retriever_valid_idx_++;
          if (retriever_valid_idx_ == valid_frames.size()) {
            // Move to next decode args
            retriever_data_idx_ += 1;
            retriever_valid_idx_ = 0;
          }
          if (retriever_data_idx_ < encoded_data_.size()) {
            next_frame_.store(encoded_data_[retriever_data_idx_].valid_frames(
                                  retriever_valid_idx_),
                              std::memory_order_release);
          }
          // printf("got frame %d\n", frames_retrieved_.load());
          total_frames_used++;
          frames_retrieved_++;
        } else {
          more_frames = decoder_->discard_frame();
        }
        current_frame_++;
        total_frames_decoded++;
      }
    }
    std::this_thread::yield();
  }
  reset_current_frame_ = -1;
  decoder_->wait_until_frames_copied();
}

void DecoderAutomata::feeder() {
  // printf("feeder start\n");
  i64 total_frames_fed = 0;
  i32 frames_fed = 0;
  seeking_ = false;
  while (not_done_) {
    {
      // Wait until frames are being requested
      std::unique_lock<std::mutex> lk(feeder_mutex_);
      feeder_waiting_ = true;
    }
    wake_feeder_.notify_one();

    {
      std::unique_lock<std::mutex> lk(feeder_mutex_);
      wake_feeder_.wait(lk, [this] { return !feeder_waiting_; });
    }
    std::atomic_thread_fence(std::memory_order_acquire);

    // Ignore requests to feed if we have alredy fed all data
    if (encoded_data_.size() <= feeder_data_idx_) {
      continue;
    }

    frames_fed = 0;
    bool seen_metadata = false;
    while (frames_retrieved_ < frames_to_get_) {
      // if (next_frame_ > feeder_next_keyframe_) {
      //   // Jump to the next
      // }
      i32 frames_to_wait = seeking_ ? 0 : 8;
      while (frames_retrieved_ < frames_to_get_ &&
            decoder_->decoded_frames_buffered() > frames_to_wait) {
        std::this_thread::yield();
      }
      if (seeking_) {
        i64 reset = encoded_data_[feeder_data_idx_].keyframes(0);
        while (reset_current_frame_ != -1 || reset > next_frame_) {
          std::this_thread::yield();
        }
        if (frames_retrieved_ >= frames_to_get_) {
          continue;
        }
        reset_current_frame_ = reset;
        seeking_ = false;
      }
      frames_fed++;

      i32 fdi = feeder_data_idx_.load(std::memory_order_acquire);
      const u8 *encoded_buffer =
          (const u8 *)encoded_data_[fdi].mutable_encoded_video()->data();
      size_t encoded_buffer_size =
          encoded_data_[fdi].mutable_encoded_video()->size();
      i32 encoded_packet_size = 0;
      const u8 *encoded_packet = NULL;
      if (feeder_buffer_offset_ < encoded_buffer_size) {
        encoded_packet_size = *reinterpret_cast<const i32 *>(
            encoded_buffer + feeder_buffer_offset_);
        feeder_buffer_offset_ += sizeof(i32);
        encoded_packet = encoded_buffer + feeder_buffer_offset_;
        assert(encoded_packet_size < encoded_buffer_size);
        feeder_buffer_offset_ += encoded_packet_size;
        // printf("encoded packet size %d, ptr %p\n", encoded_packet_size,
        //        encoded_packet);
      }

      if (seen_metadata && encoded_packet_size > 0) {
        const u8 *start_buffer = encoded_packet;
        i32 original_size = encoded_packet_size;

        while (encoded_packet_size > 0) {
          const u8 *nal_start;
          i32 nal_size;
          next_nal(encoded_packet, encoded_packet_size, nal_start, nal_size);
          if (encoded_packet_size == 0) {
            break;
          }
          i32 nal_type = get_nal_unit_type(nal_start);
          if (is_vcl_nal(nal_type)) {
            encoded_packet = nal_start -= 3;
            encoded_packet_size = nal_size + encoded_packet_size + 3;
            break;
          }
        }
      }

      //if (encoded_packet_size != 0) {
        decoder_->feed(encoded_packet, encoded_packet_size, false);
        //}
      // Set a discontinuity if we sent an empty packet to reset
      // the stream next time
      if (encoded_packet_size == 0) {
        assert(feeder_buffer_offset_ >= encoded_buffer_size);
        // Reached the end of a decoded segment so wait for decoder to flush
        // before moving onto next segment
        seen_metadata = false;
        feeder_data_idx_ += 1;
        feeder_buffer_offset_ = 0;
        seeking_ = true;
        if (encoded_data_.size() <= feeder_data_idx_) {
          break;
        }
        feeder_next_keyframe_ = encoded_data_[feeder_data_idx_].keyframes(1);
      } else {
        seen_metadata = true;
      }
      std::this_thread::yield();
    }
  }
}
}
}
