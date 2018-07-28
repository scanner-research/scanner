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

#include "scanner/util/common.h"
#include "storehouse/storage_backend.h"

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavfilter/avfilter.h"
#include "libavformat/avformat.h"
#include "libavformat/avio.h"
#include "libavutil/error.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libswscale/swscale.h"
}

using storehouse::RandomReadFile;

namespace scanner {

struct FFStorehouseState {
  std::unique_ptr<RandomReadFile> file = nullptr;
  u64 size = 0;  // total file size
  u64 pos = 0;

  u64 buffer_start = 0;
  u64 buffer_end = 0;
  std::vector<u8> buffer;
};

bool ffmpeg_storehouse_state_init(FFStorehouseState* file_state,
                                  storehouse::StorageBackend* storage,
                                  const std::string& path,
                                  std::string& error_message);

// For custom AVIOContext that loads from memory
i32 ffmpeg_storehouse_read_packet(void* opaque, u8* buf, i32 buf_size);

i64 ffmpeg_storehouse_seek(void* opaque, i64 offset, i32 whence);

}  // namespace scanner
