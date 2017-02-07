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

struct GetBitsState {
  const u8* buffer;
  i64 offset;
  i64 size;
};

inline u32 get_bit(GetBitsState &gb) {
  i32 v =
      ((*(gb.buffer + (gb.offset >> 0x3))) >> (0x7 - (gb.offset & 0x7))) & 0x1;
  gb.offset++;
  return v;
}

inline u32 get_bits(GetBitsState &gb, i32 bits) {
  i32 v = 0;
  for (i32 i = 0; i < bits; i++) {
    v |= get_bit(gb) << i;
  }
  return v;
}

inline u32 get_ue_golomb(GetBitsState &gb) {
  // calculate zero bits. Will be optimized.
  i32 zeros = 0;
  while (0 == get_bit(gb)) {
    zeros++;
  }

  // insert first 1 bit
  u32 info = 1 << zeros;

  for (i32 i = zeros - 1; i >= 0; i--) {
    info |= get_bit(gb) << i;
  }

  return (info - 1);
}


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

inline bool is_first_vcl_nal(i32 nal_type) {
  return nal_type >= 1 && nal_type <= 5;
}

struct SliceHeader {
};

inline SliceHeader parse_slice_header(GetBitsState &gb) {
  SliceHeader info;
  return info;
}

struct SPS {
  u32 sps_id;
  u32 log2_max_frame_num;
};

inline SPS parse_sps(GetBitsState &gb) {
  SPS info;
  // profile_idc
  get_bits(gb, 8);
  // constraint_set0_flag
  get_bits(gb, 1);
  // constraint_set1_flag
  get_bits(gb, 1);
  // constraint_set2_flag
  get_bits(gb, 1);
  // reserved_zero_5bits /* equal to 0 */
  get_bits(gb, 5);
  // level_idc
  get_bits(gb, 8);
  // seq_parameter_set_id
  info.sps_id = get_ue_golomb(gb);
  // log2_max_frame_num_minus4
  info.log2_max_frame_num = get_ue_golomb(gb) + 4;
  return info;
}


struct PPS {
  u32 pps_id;
  u32 sps_id;
  bool redundant_pic_cnt_present_flag;
};

inline PPS parse_pps(GetBitsState &gb) {
  PPS info;
  // pic_parameter_set_id
  info.pps_id = get_ue_golomb(gb);
  // seq_parameter_set_id
  info.sps_id = get_ue_golomb(gb);
  // entropy_coding_mode_flag
  bool entropy_coding_mode_flag = get_bit(gb);
  // pic_order_present_flag
  bool pic_order_present_flag = get_bit(gb);
  // num_slice_groups_minus1
  u32 num_slice_groups_minus1 = get_ue_golomb(gb);
  if (num_slice_groups_minus1 > 0) {
    // slice_group_map_type
    u32 slice_group_map_type = get_ue_golomb(gb);
    // FMO not supported
    assert(false);
  }
  // num_ref_idx_l0_active_minus1
  u32 num_ref_idx_l0_active_minus1 = get_ue_golomb(gb);
  // num_ref_idx_l1_active_minus1
  u32 num_ref_idx_l1_active_minus1 = get_ue_golomb(gb);
  // weighted_pred_flag
  bool weighted_pred_flag = get_bit(gb);
  // weighted_bipred_idc
  bool weighted_bipred_idc = get_bits(gb, 2);
  // pic_init_qp_minus26 /* relative to 26 */
  // HACK(apoms): should be se_golomb
  u32 pic_init_qp_minus26 = get_ue_golomb(gb);
  // pic_init_qs_minus26 /* relative to 26 */
  // HACK(apoms): should be se_golomb
  u32 pic_init_qs_minus26 = get_ue_golomb(gb);
  // chroma_qp_index_offset
  // HACK(apoms): should be se_golomb
  u32 chroma_qp_index_offset = get_ue_golomb(gb);
  // deblocking_filter_control_present_flag
  (void) get_bit(gb);
  // constrained_intra_pred_flag
  (void) get_bit(gb);
  // redundant_pic_cnt_present_flag
  info.redundant_pic_cnt_present_flag = get_bit(gb);
  // rbsp_trailing_bits()

  return info;
}
}
