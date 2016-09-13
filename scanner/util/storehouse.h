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

#include "storehouse/storage_backend.h"

#include <string>
#include <cassert>

namespace scanner {

inline void write(
  storehouse::WriteFile* file,
  const char* buffer,
  size_t size)
{
  storehouse::StoreResult result;
  EXP_BACKOFF(file->append(size, buffer), result);
  exit_on_error(result);
}

template <typename T>
inline void write(storehouse::WriteFile* file, const T& value) {
  write(file, reinterpret_cast<const char*>(&value), sizeof(T));
}

template <>
inline void write(storehouse::WriteFile* file, const std::string& s) {
  write(file, s.c_str(), s.size());
}

inline void read(
  storehouse::RandomReadFile* file,
  char* buffer,
  size_t size,
  uint64_t& pos)
{
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
inline T read(storehouse::RandomReadFile* file, uint64_t& pos) {
  T var;
  read(file, reinterpret_cast<char*>(&var), sizeof(T), pos);
  return var;
}

template <>
inline std::string read(storehouse::RandomReadFile* file, uint64_t& pos) {
  uint64_t curr_pos = pos;

  std::string var;
  while (true) {
    const size_t buf_size = 256;
    char buf[buf_size];
    read(file, buf, buf_size, pos);

    size_t buf_pos = 0;
    while (buf_pos < buf_size) {
      if (buf[buf_pos] == '\0') break;
      var += buf[buf_pos];
    }
    if (buf[buf_pos] == '\0') break;

    curr_pos += buf_size;
  }
  pos += var.size() + 1;
  return var;
}

}
