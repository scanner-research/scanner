/* Copyright 2016 Carnegie Mellon University, NVIDIA Corporation
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

#include "scanner/evaluators/video/decoder_evaluator.h"

#include "scanner/util/memory.h"

namespace scanner {

DecoderEvaluator::DecoderEvaluator(const EvaluatorConfig &config,
                                   DeviceType device_type,
                                   VideoDecoderType decoder_type)
    : device_type_(device_type), device_id_(config.device_ids[0]),
      decoder_type_(decoder_type) {
  decoder_.reset(VideoDecoder::make_from_config(device_type_, device_id_,
                                                decoder_type_, device_type_));
  assert(decoder_.get());
}

void DecoderEvaluator::configure(const VideoMetadata& metadata) {
  metadata_ = metadata;
  frame_size_ = metadata_.width() * metadata_.height() * 3;
}

void DecoderEvaluator::reset() {
  needs_warmup_ = true;
}

void DecoderEvaluator::evaluate(
  const std::vector<std::vector<u8*>>& input_buffers,
  const std::vector<std::vector<size_t>>& input_sizes,
  std::vector<std::vector<u8*>>& output_buffers,
  std::vector<std::vector<size_t>>& output_sizes)
{
  auto start = now();

  DecodeArgs& args = reinterpret_cast<DecodeArgs*>(input_buffers[1][0]);

  const u8* encoded_buffer = input_buffers[0][0];
  size_t encoded_buffer_size = input_sizes[0][0];

  bool discontinuity = needs_warmup_;
  i32 discard_until_frame =
      needs_warmup_ ? args.warmup_start_frame : args.start_frame;
  i32 total_output_frames = args.end_frame - discard_until_frame;

  size_t decoded_buffer_size = frame_size_ * total_output_frames;
  u8* decoded_buffer =
      new_buffer(device_type_, device_id_, decoded_buffer_size);

  size_t encoded_buffer_offset = 0;
  i32 current_frame = args.start_keyframe;
  while (current_frame < args.end_frame) {
    auto video_start = now();

    i32 encoded_packet_size = 0;
    u8 *encoded_packet = NULL;
    if (encoded_buffer_offset < encoded_buffer_size) {
      encoded_packet_size =
          *reinterpret_cast<i32 *>(encoded_buffer + encoded_buffer_offset);
      encoded_buffer_offset += sizeof(i32);
      encoded_packet = encoded_buffer + encoded_buffer_offset;
      encoded_buffer_offset += encoded_packet_size;
    }

    if (decoder->feed(encoded_packet, encoded_packet_size, discontinuity)) {
      // New frames
      bool more_frames = true;
      while (more_frames && current_frame < args.end_frame) {
        if (current_frame >= discard_until_frame) {
          size_t frames_buffer_offset =
              frame_size_ * (current_frame - discard_until_frame);
          assert(frames_buffer_offset < decoded_buffer_size);
          u8* current_frame_buffer_pos = decoded_buffer + frames_buffer_offset;

          more_frames =
              decoder->get_frame(current_frame_buffer_pos, frame_size_);
        } else {
          more_frames = decoder->discard_frame();
        }
        current_frame++;
      }
    }
    discontinuity = false;
  }
  // Wait on all memcpys from frames to be done
  decoder->wait_until_frames_copied();

  if (decoder->decoded_frames_buffered() > 0) {
    while (decoder->discard_frame()) {
    };
  }

  needs_warmup_ = false;

  if (profiler_) {
    profiler_->add_interval("decode", start, now());
  }
}

DecoderEvaluatorFactory::DecoderEvaluatorFactory(DeviceType device_type,
                                                 VideoDecoderType decoder_type)
    : device_type_(device_type), decoder_type_(decoder_type) {}

EvaluatorCapabilities DecoderEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = device_type_;
  caps.max_devices = 1;
  caps.warmup_size = 0;
  return caps;
}

std::vector<std::string> DecoderEvaluatorFactory::get_output_names() {
  return {"frame"};
}

Evaluator*
DecoderEvaluatorFactory::new_evaluator(const EvaluatorConfig& config) {
  return new DecoderEvaluator(config, device_type_, decoder_type_);
}
}
