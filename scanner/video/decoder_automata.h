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

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

namespace scanner {
namespace internal {

class DecoderAutomata {
  DecoderAutomata() = delete;
  DecoderAutomata(const DecoderAutomata&) = delete;
  DecoderAutomata(const DecoderAutomata&& other) = delete;

 public:
  DecoderAutomata(DeviceHandle device_handle, i32 num_devices,
                  VideoDecoderType decoder_type);
  ~DecoderAutomata();

  void initialize(const std::vector<proto::DecodeArgs>& encoded_data);

  void get_frames(u8* buffer, i32 num_frames);

  void set_profiler(Profiler* profiler);

 private:
  void feeder();

  void set_feeder_idx(i32 data_idx);

  const i32 MAX_BUFFERED_FRAMES = 8;

  Profiler* profiler_ = nullptr;

  DeviceHandle device_handle_;
  i32 num_devices_;
  VideoDecoderType decoder_type_;
  std::unique_ptr<VideoDecoder> decoder_;
  std::atomic<bool> feeder_waiting_;
  std::thread feeder_thread_;
  std::atomic<bool> not_done_;

  FrameInfo info_{};
  size_t frame_size_;
  i32 current_frame_;
  std::atomic<i32> reset_current_frame_;
  std::vector<proto::DecodeArgs> encoded_data_;

  std::atomic<i64> next_frame_;
  std::atomic<i64> frames_retrieved_;
  std::atomic<i64> frames_to_get_;

  std::atomic<i32> retriever_data_idx_;
  std::atomic<i32> retriever_valid_idx_;

  std::atomic<bool> skip_frames_;
  std::atomic<bool> seeking_;
  std::atomic<i32> feeder_data_idx_;
  std::atomic<i64> feeder_valid_idx_;
  std::atomic<i64> feeder_current_frame_;
  std::atomic<i64> feeder_next_frame_;

  std::atomic<size_t> feeder_buffer_offset_;
  std::atomic<i64> feeder_next_keyframe_;
  std::mutex feeder_mutex_;
  std::condition_variable wake_feeder_;
};
}
}
