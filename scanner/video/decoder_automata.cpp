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

#include "scanner/util/memory.h"
#include "scanner/util/h264.h"

#include <thread>

namespace scanner {
namespace internal {

DecoderAutomata::DecoderAutomata(DeviceHandle device_handle, i32 num_devices,
                                 VideoDecoderType decoder_type)
    : device_handle_(device_handle), num_devices_(num_devices),
      decoder_type_(decoder_type),
      decoder_(VideoDecoder::make_from_config(device_handle, num_devices,
                                              decoder_type)),
      feeder_thread_(&DecoderAutomata::feeder, this),
      not_done_(true) {}

DecoderAutomata::~DecoderAutomata() {
  {
    std::unique_lock<std::mutex> lk(feeder_mutex_);
    frames_retrieved_ = 0;
    frames_to_get_ = 0;
    not_done_ = false;
    wake_feeder_.notify_one();
  }
  feeder_thread_.join();
}

void DecoderAutomata::initialize(
    const std::vector<proto::DecodeArgs *> &encoded_data) {
  assert(!encoded_data.empty());
  frame_size_ = encoded_data[0]->width() * encoded_data[0]->height() * 3;
  next_frame_.store(encoded_data[0]->start_keyframe(),
                    std::memory_order_release);
  retriever_data_idx_.store(0, std::memory_order_release);
  retriever_valid_idx_ = 0;
  feeder_data_idx_.store(0, std::memory_order_release);
  feeder_buffer_offset_.store(0, std::memory_order_release);
  feeder_next_keyframe_.store(encoded_data[0]->keyframes(1),
                              std::memory_order_release);
  encoded_data_ = encoded_data;
}

void DecoderAutomata::get_frames(u8* buffer, i32 num_frames) {
  i64 total_frames_decoded = 0;
  i64 total_frames_used = 0;

  // profiler_->add_interval("decode", start, now());
  // profiler_->increment("effective_frames", total_frames_used);
  // profiler_->increment("decoded_frames", total_frames_decoded);

  auto start = now();

  // Start up feeder thread
  {
    std::unique_lock<std::mutex> lk(feeder_mutex_);
    frames_retrieved_ = 0;
    frames_to_get_ = num_frames;
    wake_feeder_.notify_one();
  }

  while (frames_retrieved_ < frames_to_get_) {
    if (decoder_->decoded_frames_buffered() > 0) {
      // New frames
      bool more_frames = true;
      while (more_frames && frames_retrieved_ < frames_to_get_) {
        const auto &valid_frames =
            encoded_data_[retriever_data_idx_]->valid_frames();
        assert(current_frame_ <= valid_frames.Get(retriever_valid_idx_));
        if (current_frame_ == valid_frames.Get(retriever_valid_idx_)) {
          u8 *decoded_buffer = buffer + frames_retrieved_ * frame_size_;
          more_frames = decoder_->get_frame(decoded_buffer, frame_size_);
          retriever_valid_idx_++;
          if (retriever_valid_idx_ == valid_frames.size()) {
            // Move to next decode args
            retriever_data_idx_ += 1;
            retriever_valid_idx_ = 0;
          }
          next_frame_.store(encoded_data_[retriever_data_idx_]->valid_frames(
                                retriever_valid_idx_),
                            std::memory_order_release);
          total_frames_used++;
        } else {
          more_frames = decoder_->discard_frame();
        }
        current_frame_++;
        total_frames_decoded++;
      }
    }
    std::this_thread::yield();
  }
  decoder_->wait_until_frames_copied();
}

void DecoderAutomata::feeder() {
  while (!not_done_) {
    {
      // Wait until frames are being requested
      std::unique_lock<std::mutex> lk(feeder_mutex_);
      wake_feeder_.wait(lk, [this] { return frames_retrieved_ == 0; });
    }

    bool saw_end_packet = false;
    bool seen_metadata = false;
    while (frames_retrieved_ < frames_to_get_) {
      const u8 *encoded_buffer =
          (const u8 *)
              encoded_data_[feeder_data_idx_.load(std::memory_order_acquire)]
                  ->mutable_encoded_video()
                  ->data();
      size_t encoded_buffer_size =
          encoded_data_[feeder_data_idx_.load(std::memory_order_acquire)]
              ->mutable_encoded_video()
              ->size();
      i32 encoded_packet_size = 0;
      const u8 *encoded_packet = NULL;
      if (feeder_buffer_offset_ < encoded_buffer_size) {
        encoded_packet_size = *reinterpret_cast<const i32 *>(
            encoded_buffer + feeder_buffer_offset_);
        feeder_buffer_offset_ += sizeof(i32);
        encoded_packet = encoded_buffer + feeder_buffer_offset_;
        feeder_buffer_offset_ += encoded_packet_size;
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
        assert(encoded_packet_size != 0);
      }

      decoder_->feed(encoded_packet, encoded_packet_size, false);
      seen_metadata = true;
      // Set a discontinuity if we sent an empty packet to reset
      // the stream next time
      saw_end_packet = (encoded_packet_size == 0);
      if (encoded_packet_size == 0) {
        assert(feeder_buffer_offset_ >= encoded_buffer_size);
        break;
      }
    }
  }
}

}
}
