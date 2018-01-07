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

#include "scanner/util/common.h"
#include "storehouse/storage_backend.h"

#include <cassert>
#include <string>

namespace scanner {

inline void s_write(storehouse::WriteFile* file, const u8* buffer,
                    size_t size) {
  storehouse::StoreResult result;
  EXP_BACKOFF(file->append(size, buffer), result);
  exit_on_error(result);
}

template <typename T>
inline void s_write(storehouse::WriteFile* file, const T& value) {
  s_write(file, reinterpret_cast<const u8*>(&value), sizeof(T));
}

template <>
inline void s_write(storehouse::WriteFile* file, const std::string& s) {
  s_write(file, reinterpret_cast<const u8*>(s.c_str()), s.size() + 1);
}

inline void s_read(storehouse::RandomReadFile* file, u8* buffer, size_t size,
                   u64& pos) {
  VLOG(2) << "Reading " << file->path() << " (size " << size << ", pos " << pos
          << ")";
  storehouse::StoreResult result;
  size_t size_read;
  EXP_BACKOFF(file->read(pos, size, buffer, size_read), result);
  if (result != storehouse::StoreResult::EndOfFile) {
    exit_on_error(result);
  }
  assert(size_read == size);
  pos += size_read;
}

template <typename T>
inline T s_read(storehouse::RandomReadFile* file, u64& pos) {
  T var;
  s_read(file, reinterpret_cast<u8*>(&var), sizeof(T), pos);
  return var;
}

template <>
inline std::string s_read(storehouse::RandomReadFile* file, u64& pos) {
  u64 curr_pos = pos;

  std::string var;
  while (true) {
    const size_t buf_size = 256;
    u8 buf[buf_size];

    storehouse::StoreResult result;
    size_t size_read;
    EXP_BACKOFF(file->read(pos, buf_size, buf, size_read), result);
    if (result != storehouse::StoreResult::EndOfFile) {
      exit_on_error(result);
      assert(size_read == buf_size);
    }

    size_t buf_pos = 0;
    while (buf_pos < buf_size) {
      if (buf[buf_pos] == '\0') break;
      var += buf[buf_pos];
      buf_pos++;
    }
    if (buf_pos < buf_size && buf[buf_pos] == '\0') break;

    curr_pos += buf_size;
  }
  pos += var.size() + 1;
  return var;
}
}
