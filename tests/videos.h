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

#include "scanner/util/fs.h"
#include "scanner/video/decoder_automata.h"

#include <thread>

namespace scanner {
struct TestVideoInfo {
  TestVideoInfo(i32 w, i32 h, const std::string& u, const std::string& m)
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

const TestVideoInfo long_video(
    640, 480,
    "https://storage.googleapis.com/scanner-data/test/long_video.h264",
    "https://storage.googleapis.com/scanner-data/test/long_video_meta.bin");

inline std::string download_video(const TestVideoInfo& info) {
  std::string local_video_path;
  temp_file(local_video_path);
  download(info.data_url, local_video_path);
  return local_video_path;
}

inline std::string download_video_meta(const TestVideoInfo& info) {
  std::string local_path;
  temp_file(local_path);
  download(info.metadata_url, local_path);
  return local_path;
}
}
