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
  : device_handle_(device_handle),
    num_devices_(num_devices),
    decoder_type_(decoder_type),
    decoder_(
      VideoDecoder::make_from_config(device_handle, num_devices, decoder_type)),
    feeder_waiting_(false),
    not_done_(true),
    frames_retrieved_(0),
    skip_frames_(false) {
  feeder_thread_ = std::thread(&DecoderAutomata::feeder, this);
}

DecoderAutomata::~DecoderAutomata() {
  {
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
  const std::vector<proto::DecodeArgs>& encoded_data) {
  assert(!encoded_data.empty());
  encoded_data_ = encoded_data;
  frame_size_ = encoded_data[0].width() * encoded_data[0].height() * 3;
  current_frame_ = encoded_data[0].start_keyframe();
  next_frame_.store(encoded_data[0].valid_frames(0), std::memory_order_release);
  retriever_data_idx_.store(0, std::memory_order_release);
  retriever_valid_idx_ = 0;

  FrameInfo info(encoded_data[0].height(), encoded_data[0].width(), 3,
                 FrameType::U8);

  while (decoder_->discard_frame()) {
  }

  std::unique_lock<std::mutex> lk(feeder_mutex_);
  wake_feeder_.wait(lk, [this] { return feeder_waiting_.load(); });

  if (info_ != info) {
    decoder_->configure(info);
  }
  if (frames_retrieved_ > 0) {
    decoder_->feed(nullptr, 0, true);
  }

  set_feeder_idx(0);
  info_ = info;
  std::atomic_thread_fence(std::memory_order_release);
  seeking_ = false;
}

void DecoderAutomata::get_frames(u8* buffer, i32 num_frames) {
  i64 total_frames_decoded = 0;
  i64 total_frames_used = 0;

  auto start = now();

  // Wait until feeder is waiting
  {
    // Wait until frames are being requested
    std::unique_lock<std::mutex> lk(feeder_mutex_);
    wake_feeder_.wait(lk, [this] { return feeder_waiting_.load(); });
  }

  if (encoded_data_.size() > feeder_data_idx_) {
    // Make sure to not feed seek packet if we reached end of stream
    if (seeking_) {
      decoder_->feed(nullptr, 0, true);
      seeking_ = false;
    }
  }

  // Start up feeder thread
  {
    std::unique_lock<std::mutex> lk(feeder_mutex_);
    frames_retrieved_ = 0;
    frames_to_get_ = num_frames;
    feeder_waiting_ = false;
  }
  wake_feeder_.notify_one();

  if (profiler_) {
    profiler_->add_interval("get_frames_wait", start, now());
  }

  while (frames_retrieved_ < frames_to_get_) {
    if (decoder_->decoded_frames_buffered() > 0) {
      auto iter = now();
      // New frames
      bool more_frames = true;
      while (more_frames && frames_retrieved_ < frames_to_get_) {
        const auto& valid_frames =
          encoded_data_[retriever_data_idx_].valid_frames();
        assert(valid_frames.size() > retriever_valid_idx_.load());
        assert(current_frame_ <= valid_frames.Get(retriever_valid_idx_));
        if (current_frame_ == valid_frames.Get(retriever_valid_idx_)) {
          u8* decoded_buffer = buffer + frames_retrieved_ * frame_size_;
          more_frames = decoder_->get_frame(decoded_buffer, frame_size_);
          retriever_valid_idx_++;
          if (retriever_valid_idx_ == valid_frames.size()) {
            // Move to next decode args
            retriever_data_idx_ += 1;
            retriever_valid_idx_ = 0;

            // Trigger feeder to start again and set ourselves to the
            // start of that keyframe
            if (retriever_data_idx_ < encoded_data_.size()) {
              {
                // Wait until feeder is waiting
                // skip_frames_ = true;
                std::unique_lock<std::mutex> lk(feeder_mutex_);
                wake_feeder_.wait(lk, [this, &total_frames_decoded] {
                  while (decoder_->discard_frame()) {
                    total_frames_decoded++;
                  }
                  return feeder_waiting_.load();
                });
                // skip_frames_ = false;
              }

              if (seeking_) {
                decoder_->feed(nullptr, 0, true);
                seeking_ = false;
              }

              {
                std::unique_lock<std::mutex> lk(feeder_mutex_);
                feeder_waiting_ = false;
                current_frame_ =
                  encoded_data_[retriever_data_idx_].keyframes(0) - 1;
              }
              wake_feeder_.notify_one();
              more_frames = false;
            } else {
              assert(frames_retrieved_ + 1 == frames_to_get_);
            }
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
        // printf("curr frame %d, frames decoded %d\n", current_frame_,
        //        total_frames_decoded);
      }
      if (profiler_) {
        profiler_->add_interval("iter", iter, now());
      }
    }
    std::this_thread::yield();
  }
  decoder_->wait_until_frames_copied();
  if (profiler_) {
    profiler_->add_interval("get_frames", start, now());
    profiler_->increment("frames_used", total_frames_used);
    profiler_->increment("frames_decoded", total_frames_decoded);
  }
}

void DecoderAutomata::set_profiler(Profiler* profiler) {
  profiler_ = profiler;
  decoder_->set_profiler(profiler);
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

    if (profiler_) {
      profiler_->increment("frames_fed", frames_fed);
    }
    frames_fed = 0;
    bool seen_metadata = false;
    while (frames_retrieved_ < frames_to_get_) {
      i32 frames_to_wait = 8;
      while (frames_retrieved_ < frames_to_get_ &&
             decoder_->decoded_frames_buffered() > frames_to_wait) {
        wake_feeder_.notify_one();
        std::this_thread::yield();
      }
      if (skip_frames_) {
        seen_metadata = false;
        seeking_ = true;
        set_feeder_idx(feeder_data_idx_ + 1);
        break;
      }
      frames_fed++;

      i32 fdi = feeder_data_idx_.load(std::memory_order_acquire);
      const u8* encoded_buffer =
        (const u8*)encoded_data_[fdi].mutable_encoded_video()->data();
      size_t encoded_buffer_size =
        encoded_data_[fdi].mutable_encoded_video()->size();
      i32 encoded_packet_size = 0;
      const u8* encoded_packet = NULL;
      if (feeder_buffer_offset_ < encoded_buffer_size) {
        encoded_packet_size =
          *reinterpret_cast<const i32*>(encoded_buffer + feeder_buffer_offset_);
        feeder_buffer_offset_ += sizeof(i32);
        encoded_packet = encoded_buffer + feeder_buffer_offset_;
        assert(encoded_packet_size < encoded_buffer_size);
        feeder_buffer_offset_ += encoded_packet_size;
        // printf("encoded packet size %d, ptr %p\n", encoded_packet_size,
        //        encoded_packet);
      }

      if (seen_metadata && encoded_packet_size > 0) {
        const u8* start_buffer = encoded_packet;
        i32 original_size = encoded_packet_size;

        while (encoded_packet_size > 0) {
          const u8* nal_start;
          i32 nal_size;
          next_nal(encoded_packet, encoded_packet_size, nal_start, nal_size);
          if (encoded_packet_size == 0) {
            break;
          }
          i32 nal_type = get_nal_unit_type(nal_start);
          i32 nal_ref = get_nal_ref_idc(nal_start);
          if (is_vcl_nal(nal_type)) {
            encoded_packet = nal_start -= 3;
            encoded_packet_size = nal_size + encoded_packet_size + 3;
            break;
          }
        }
      }

      decoder_->feed(encoded_packet, encoded_packet_size, false);

      if (feeder_current_frame_ == feeder_next_frame_) {
        feeder_valid_idx_++;
        if (feeder_valid_idx_ <
            encoded_data_[feeder_data_idx_].valid_frames_size()) {
          feeder_next_frame_ =
            encoded_data_[feeder_data_idx_].valid_frames(feeder_valid_idx_);
        } else {
          // Done
          feeder_next_frame_ = -1;
        }
      }
      feeder_current_frame_++;

      // Set a discontinuity if we sent an empty packet to reset
      // the stream next time
      if (encoded_packet_size == 0) {
        assert(feeder_buffer_offset_ >= encoded_buffer_size);
        // Reached the end of a decoded segment so wait for decoder to flush
        // before moving onto next segment
        seen_metadata = false;
        seeking_ = true;
        set_feeder_idx(feeder_data_idx_ + 1);
        break;
      } else {
        seen_metadata = true;
      }
      std::this_thread::yield();
    }
  }
}

void DecoderAutomata::set_feeder_idx(i32 data_idx) {
  feeder_data_idx_ = data_idx;
  feeder_valid_idx_ = 0;
  feeder_buffer_offset_ = 0;
  if (feeder_data_idx_ < encoded_data_.size()) {
    feeder_current_frame_ = encoded_data_[feeder_data_idx_].keyframes(0);
    feeder_next_frame_ = encoded_data_[feeder_data_idx_].valid_frames(0);
    feeder_next_keyframe_ = encoded_data_[feeder_data_idx_].keyframes(1);
  }
}
}
}
