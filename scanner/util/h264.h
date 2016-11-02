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

namespace scanner {

inline void next_nal(const u8*& buffer, i32& buffer_size_left,
                     const u8*& nal_start, i32& nal_size) {
  while (buffer_size_left > 2 &&
         !(buffer[0] == 0x00 && buffer[1] == 0x00 && buffer[2] == 0x01)) {
    buffer++;
    buffer_size_left--;
  }

  buffer += 3;
  buffer_size_left -= 3;

  nal_start = buffer;
  nal_size = 0;
  if (buffer_size_left > 2) {
    while (!(buffer[0] == 0x00 && buffer[1] == 0x00 &&
             (buffer[2] == 0x00 || buffer[2] == 0x01))) {
      buffer++;
      buffer_size_left--;
      nal_size++;
      if (buffer_size_left < 3) {
        nal_size += buffer_size_left;
        break;
      }
    }
  }
}

inline i32 get_nal_unit_type(const u8* nal_start) {
  return (*nal_start) & 0x1F;
}

inline i32 get_nal_ref_idc(const u8* nal_start) { return (*nal_start >> 5); }

inline bool is_vcl_nal(i32 nal_type) { return nal_type >= 1 && nal_type <= 5; }

inline i32 get_bit(const u8* const base, i32 offset) {
  return ((*(base + (offset >> 0x3))) >> (0x7 - (offset & 0x7))) & 0x1;
}

inline i32 parse_exp_golomb(const u8*& buffer, i32& size, i32& offset) {
  // calculate zero bits. Will be optimized.
  i32 zeros = 0;
  while (0 == get_bit(buffer, offset++)) {
    zeros++;
  }

  // insert first 1 bit
  i32 info = 1 << zeros;

  for (i32 i = zeros - 1; i >= 0; i--) {
    info |= get_bit(buffer, offset++) << i;
  }

  return (info - 1);
}
}
