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
#include "scanner/util/fs.h"

#include <gtest/gtest.h>

#include <thread>

namespace scanner {
namespace internal {

struct TestVideoInfo {
  TestVideoInfo(i32 w, i32 h, const std::string &u, const std::string& m)
      : width(w), height(h), data_url(u), metadata_url(m) {}

  i32 width;
  i32 height;
  std::string data_url;
  std::string metadata_url;
};

const TestVideoInfo short_video(
    640, 480,
    "https://storage.googleapis.com/scanner-data/test/short_video.h264",
    "https://storage.googleapis.com/scanner-data/test/short_video_meta.bin");

std::string download_video(const TestVideoInfo& info) {
  std::string local_video_path;
  temp_file(local_video_path);
  download(info.data_url, local_video_path);
  return local_video_path;
}

std::string download_video_meta(const TestVideoInfo& info) {
  std::string local_path;
  temp_file(local_path);
  download(info.metadata_url, local_path);
  return local_path;
}

TEST(DecoderAutomata, GetAllFrames) {
  std::unique_ptr<storehouse::StorageConfig> sc(
      storehouse::StorageConfig::make_posix_config());
  auto storage = storehouse::StorageBackend::make_from_config(sc.get());
  VideoDecoderType decoder_type = VideoDecoderType::SOFTWARE;
  DeviceHandle device = CPU_DEVICE;
  DecoderAutomata* decoder = new DecoderAutomata(device, 1, decoder_type);

  // Load test data
  VideoMetadata video_meta =
      read_video_metadata(storage, download_video_meta(short_video));
  std::vector<u8> video_bytes = read_entire_file(download_video(short_video));

  std::vector<proto::DecodeArgs> args;
  args.emplace_back();
  proto::DecodeArgs& decode_args = args.back();
  decode_args.set_width(video_meta.width());
  decode_args.set_height(video_meta.height());
  decode_args.set_start_keyframe(0);
  decode_args.set_end_keyframe(video_meta.frames());
  for (i64 r = 0; r < video_meta.frames(); ++r) {
    decode_args.add_valid_frames(r);
  }
  for (i64 k : video_meta.keyframe_positions()) {
    decode_args.add_keyframes(k);
  }
  for (i64 k : video_meta.keyframe_byte_offsets()) {
    decode_args.add_keyframe_byte_offsets(k);
  }
  decode_args.set_encoded_video(video_bytes.data(), video_bytes.size());

  decoder->initialize(args);

  std::vector<u8> frame_buffer(short_video.width * short_video.height * 3);
  for (i64 i = 0; i < video_meta.frames(); ++i) {
    decoder->get_frames(frame_buffer.data(), 1);
  }

  delete decoder;
  delete storage;
}

TEST(DecoderAutomata, GetStridedFrames) {
  std::unique_ptr<storehouse::StorageConfig> sc(
      storehouse::StorageConfig::make_posix_config());
  auto storage = storehouse::StorageBackend::make_from_config(sc.get());
  VideoDecoderType decoder_type = VideoDecoderType::SOFTWARE;
  DeviceHandle device = CPU_DEVICE;
  DecoderAutomata* decoder = new DecoderAutomata(device, 1, decoder_type);

  // Load test data
  VideoMetadata video_meta =
      read_video_metadata(storage, download_video_meta(short_video));
  std::vector<u8> video_bytes = read_entire_file(download_video(short_video));

  std::vector<proto::DecodeArgs> args;
  args.emplace_back();
  proto::DecodeArgs& decode_args = args.back();
  decode_args.set_width(video_meta.width());
  decode_args.set_height(video_meta.height());
  decode_args.set_start_keyframe(0);
  decode_args.set_end_keyframe(video_meta.frames());
  for (i64 r = 0; r < video_meta.frames(); r+=2) {
    decode_args.add_valid_frames(r);
  }
  for (i64 k : video_meta.keyframe_positions()) {
    decode_args.add_keyframes(k);
  }
  for (i64 k : video_meta.keyframe_byte_offsets()) {
    decode_args.add_keyframe_byte_offsets(k);
  }
  decode_args.set_encoded_video(video_bytes.data(), video_bytes.size());

  decoder->initialize(args);

  std::vector<u8> frame_buffer(short_video.width * short_video.height * 3);
  for (i64 i = 0; i < video_meta.frames() / 2; ++i) {
    decoder->get_frames(frame_buffer.data(), 1);
  }

  delete decoder;
  delete storage;
}

}
}
