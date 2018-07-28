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

#include "scanner/util/ffmpeg.h"

using storehouse::StoreResult;

namespace scanner {

bool ffmpeg_storehouse_state_init(FFStorehouseState* file_state,
                                  storehouse::StorageBackend* storage,
                                  const std::string& path,
                                  std::string& error_message) {
  StoreResult result;
  EXP_BACKOFF(make_unique_random_read_file(storage, path, file_state->file),
              result);
  if (result != StoreResult::Success) {
    error_message = "Can not open video file";
    return false;
  }

  EXP_BACKOFF(file_state->file->get_size(file_state->size), result);
  if (result != StoreResult::Success) {
    error_message = "Can not get file size";
    return false;
  }
  if (file_state->size <= 0) {
    error_message = "Can not ingest empty video file";
    return false;
  }

  file_state->pos = 0;
  return true;
}

i32 ffmpeg_storehouse_read_packet(void* opaque, u8* buf, i32 buf_size) {
  FFStorehouseState* fs = (FFStorehouseState*)opaque;
  if (!(fs->buffer_start <= fs->pos && fs->pos + buf_size < fs->buffer_end)) {
    // Not in cache
    size_t buffer_size = 64 * 1024 * 1024;
    fs->buffer.resize(buffer_size);
    size_t size_read;
    storehouse::StoreResult result;
    EXP_BACKOFF(
        fs->file->read(fs->pos, buffer_size, fs->buffer.data(), size_read),
        result);
    if (result != storehouse::StoreResult::EndOfFile) {
      exit_on_error(result);
    }

    fs->buffer_start = fs->pos;
    fs->buffer_end = fs->pos + size_read;
  }

  size_t size_read =
      std::min((size_t)buf_size, (size_t)(fs->buffer_end - fs->pos));
  memcpy(buf, fs->buffer.data() + (fs->pos - fs->buffer_start), size_read);
  fs->pos += size_read;
  return static_cast<i32>(size_read);
}

i64 ffmpeg_storehouse_seek(void* opaque, i64 offset, i32 whence) {
  FFStorehouseState* fs = (FFStorehouseState*)opaque;
  switch (whence) {
    case SEEK_SET:
      assert(offset >= 0);
      fs->pos = static_cast<u64>(offset);
      break;
    case SEEK_CUR:
      fs->pos += offset;
      break;
    case SEEK_END:
      fs->pos = fs->size;
      break;
    case AVSEEK_SIZE:
      return fs->size;
      break;
  }
  return fs->size - fs->pos;
}

}  // namespace scanner
