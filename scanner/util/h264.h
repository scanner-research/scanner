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

inline u32 get_bit(GetBitsState& gb) {
  u8 v =
      ((*(gb.buffer + (gb.offset >> 0x3))) >> (0x7 - (gb.offset & 0x7))) & 0x1;
  gb.offset++;
  return v;
}

inline u32 get_bits(GetBitsState& gb, i32 bits) {
  u32 v = 0;
  for (i32 i = bits - 1; i >= 0; i--) {
    v |= get_bit(gb) << i;
  }
  return v;
}

inline u32 get_ue_golomb(GetBitsState& gb) {
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

inline u32 get_se_golomb(GetBitsState& gb) {
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
  bool found = false;
  while (buffer_size_left > 2) {
    if (buffer[0] == 0x00 && buffer[1] == 0x00 && buffer[2] == 0x01) {
      found = true;
      break;
    }
    buffer++;
    buffer_size_left--;
  }

  buffer += 3;
  buffer_size_left -= 3;

  nal_start = buffer;
  nal_size = 0;

  if (!found) {
    return;
  }
  while (buffer_size_left > 2 &&
         !(buffer[0] == 0x00 && buffer[1] == 0x00 &&
           (buffer[2] == 0x00 || buffer[2] == 0x01))) {
    buffer++;
    buffer_size_left--;
    nal_size++;
  }
  if (!(buffer_size_left > 3)) {
    nal_size += buffer_size_left;
    // buffer += buffer_size_left;
    // buffer_size_left = 0;
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

struct SPS {
  i32 profile_idc;
  u32 sps_id;
  u32 log2_max_frame_num;
  u32 poc_type;
  u32 log2_max_pic_order_cnt_lsb;
  bool delta_pic_order_always_zero_flag;
  bool frame_mbs_only_flag;
};

inline bool parse_sps(GetBitsState& gb, SPS& info) {
  // profile_idc
  info.profile_idc = get_bits(gb, 8);
  // constraint_set0_flag
  get_bit(gb);
  // constraint_set1_flag
  get_bit(gb);
  // constraint_set2_flag
  get_bit(gb);
  // reserved_zero_5bits /* equal to 0 */
  get_bits(gb, 5);
  // level_idc
  get_bits(gb, 8);
  // seq_parameter_set_id
  info.sps_id = get_ue_golomb(gb);
  if (info.profile_idc == 100 ||  // High profile
      info.profile_idc == 110 ||  // High10 profile
      info.profile_idc == 122 ||  // High422 profile
      info.profile_idc == 244 ||  // High444 Predictive profile
      info.profile_idc == 44 ||   // Cavlc444 profile
      info.profile_idc == 83 ||   // Scalable Constrained High profile (SVC)
      info.profile_idc == 86 ||   // Scalable High Intra profile (SVC)
      info.profile_idc == 118 ||  // Stereo High profile (MVC)
      info.profile_idc == 128 ||  // Multiview High profile (MVC)
      info.profile_idc == 138 ||  // Multiview Depth High profile (MVCD)
      info.profile_idc == 139 ||
      info.profile_idc == 134 ||
      info.profile_idc == 135 ||
      info.profile_idc == 144) {
    // chroma_format_idc
    u32 chroma_format_idc = get_ue_golomb(gb);
    if (chroma_format_idc > 3U) {
      LOG(WARNING) << "invalid chroma format idc " << chroma_format_idc;
      return false;
    } else if (chroma_format_idc == 3) {
      // residual
      bool residual_color_transform_flag = get_bit(gb);
      if (residual_color_transform_flag) {
        LOG(WARNING) << "separate color planes not supported";
        return false;
      }
    }
    // bit_depth_luma
    u32 bit_depth_luma = get_ue_golomb(gb) + 8;
    u32 bit_depth_chroma = get_ue_golomb(gb) + 8;
    if (bit_depth_chroma != bit_depth_luma) {
      LOG(WARNING) << "separate color planes not supported";
      return false;
    }
    if (bit_depth_luma < 8 || bit_depth_luma > 14 || bit_depth_chroma < 8 ||
        bit_depth_chroma > 14) {
      LOG(WARNING) << "illegal bit depth value: " << bit_depth_luma << ", "
                   << bit_depth_chroma;
      return false;
    }
    // transform_bypass
    get_bit(gb);
    // scaling_matrix
    if (get_bit(gb)) {
      LOG(WARNING) << "scaling matrix not supported";
      return false;
    }
  }
  VLOG(1) << "profile idc " << (i32)info.profile_idc;
  // log2_max_frame_num_minus4
  info.log2_max_frame_num = get_ue_golomb(gb) + 4;
  // pic_order_cnt_type
  info.poc_type = get_ue_golomb(gb);
  switch (info.poc_type) {
    case 0: {
      // log2_max_pic_order_cnt_lsb_minus4
      info.log2_max_pic_order_cnt_lsb = get_ue_golomb(gb) + 4;
    } break;
    case 1: {
      // delta_pic_order_always_zero_flag
      info.delta_pic_order_always_zero_flag = get_bit(gb);
      // offset_for_non_ref_pic
      get_se_golomb(gb);
      // offset_for_top_to_bottom_field
      get_se_golomb(gb);
      // num_ref_frames_in_pic_order_cnt_cycle
      u32 num_ref_frames = get_ue_golomb(gb);
      for (u32 i = 0; i < num_ref_frames; i++) {
        // offset_for_ref_frame[ i ];
        get_se_golomb(gb);
      }
    } break;
    case 2: {
      // NOTE(apoms): Nothing to do here
    } break;
    default: {
      LOG(WARNING) << "Illegal picture_order_count type: " << info.poc_type;
      return false;
    } break;
  }
  // num_ref_frames
  get_ue_golomb(gb);
  // gaps_in_frame_num_value_allowed_flag
  get_bit(gb);
  // pic_width_in_mbs_minus1
  get_ue_golomb(gb);
  // pic_height_in_map_units_minus1
  get_ue_golomb(gb);
  // frame_mbs_only_flag
  info.frame_mbs_only_flag = get_bit(gb);
  // TODO(apoms): parse rest of it
  return true;
}

struct PPS {
  u32 pps_id;
  u32 sps_id;
  bool pic_order_present_flag;
  bool redundant_pic_cnt_present_flag;
  u32 num_ref_idx_l0_default_active;
  u32 num_ref_idx_l1_default_active;
  bool weighted_pred_flag;
  u8 weighted_bipred_idc;
};

inline bool parse_pps(GetBitsState& gb, PPS& info) {
  // pic_parameter_set_id
  info.pps_id = get_ue_golomb(gb);
  // seq_parameter_set_id
  info.sps_id = get_ue_golomb(gb);
  // entropy_coding_mode_flag
  bool entropy_coding_mode_flag = get_bit(gb);
  // pic_order_present_flag
  info.pic_order_present_flag = get_bit(gb);
  // num_slice_groups_minus1
  u32 num_slice_groups_minus1 = get_ue_golomb(gb);
  if (num_slice_groups_minus1 > 0) {
    // slice_group_map_type
    u32 slice_group_map_type = get_ue_golomb(gb);
    // FMO not supported
    LOG(WARNING) << "FMO encoded video not supported";
    return false;
  }
  // num_ref_idx_l0_active_minus1
  info.num_ref_idx_l0_default_active = get_ue_golomb(gb) + 1;
  // num_ref_idx_l1_active_minus1
  info.num_ref_idx_l1_default_active = get_ue_golomb(gb) + 1;
  // weighted_pred_flag
  info.weighted_pred_flag = get_bit(gb);
  // weighted_bipred_idc
  info.weighted_bipred_idc = get_bits(gb, 2);
  // pic_init_qp_minus26 /* relative to 26 */
  u32 pic_init_qp_minus26 = get_se_golomb(gb);
  // pic_init_qs_minus26 /* relative to 26 */
  u32 pic_init_qs_minus26 = get_se_golomb(gb);
  // chroma_qp_index_offset
  u32 chroma_qp_index_offset = get_se_golomb(gb);
  // deblocking_filter_control_present_flag
  (void)get_bit(gb);
  // constrained_intra_pred_flag
  (void)get_bit(gb);
  // redundant_pic_cnt_present_flag
  info.redundant_pic_cnt_present_flag = get_bit(gb);
  // rbsp_trailing_bits()

  return true;
}

struct SliceHeader {
  u32 nal_unit_type;
  u32 nal_ref_idc;
  u32 slice_type;
  u32 sps_id;  // Added for convenience
  u32 pps_id;
  u32 frame_num;
  bool field_pic_flag;
  bool bottom_field_flag;
  u32 idr_pic_id;
  u32 pic_order_cnt_lsb;
  i32 delta_pic_order_cnt_bottom;
  u32 delta_pic_order_cnt[2];
  u32 redundant_pic_cnt;
  u32 num_ref_idx_l0_active;
  u32 num_ref_idx_l1_active;
};

inline bool parse_slice_header(GetBitsState& gb, SPS& sps,
                               std::map<u32, PPS>& pps_map, u32 nal_unit_type,
                               u32 nal_ref_idc, SliceHeader& info) {
  info.nal_unit_type = nal_unit_type;
  info.nal_ref_idc = nal_ref_idc;
  // first_mb_in_slice
  get_ue_golomb(gb);
  // slice_type
  info.slice_type = get_ue_golomb(gb);
  if (info.slice_type > 9) {
    LOG(WARNING) << "Slice type too long";
    return false;
  }
  info.sps_id = sps.sps_id;
  // pic_parameter_set_id
  info.pps_id = get_ue_golomb(gb);
  PPS& pps = pps_map.at(info.pps_id);
  // frame_num
  info.frame_num = get_bits(gb, sps.log2_max_frame_num);
  if (!sps.frame_mbs_only_flag) {
    // field_pic_flag
    info.field_pic_flag = get_bit(gb);
    // bottom_field_flag
    info.bottom_field_flag = info.field_pic_flag ? get_bit(gb) : 0;
  } else {
    info.field_pic_flag = 0;
    info.bottom_field_flag = -1;
  }
  if (nal_unit_type == 5) {
    // idr_pic_id
    info.idr_pic_id = get_ue_golomb(gb);
  }
  info.delta_pic_order_cnt_bottom = 0;
  if (sps.poc_type == 0) {
    // pic_order_cnt_lsb
    info.pic_order_cnt_lsb = get_bits(gb, sps.log2_max_pic_order_cnt_lsb);

    if (pps.pic_order_present_flag == 1 && !info.field_pic_flag) {
      info.delta_pic_order_cnt_bottom = get_se_golomb(gb);
    }
  }
  info.delta_pic_order_cnt[0] = 0;
  info.delta_pic_order_cnt[1] = 0;
  if (sps.delta_pic_order_always_zero_flag) {
    info.delta_pic_order_cnt[0] = 0;
    info.delta_pic_order_cnt[1] = 0;
  } else if (sps.poc_type == 1) {
    info.delta_pic_order_cnt[0] = get_se_golomb(gb);
    if ((pps.pic_order_present_flag == 1) && !info.field_pic_flag) {
      info.delta_pic_order_cnt[1] = get_se_golomb(gb);
    } else {
      info.delta_pic_order_cnt[1] = 0;
    }
  }
  info.redundant_pic_cnt =
      pps.redundant_pic_cnt_present_flag ? get_ue_golomb(gb) : 0;
  if (info.slice_type == 1 || info.slice_type == 6) {
    bool direct_spatial_mv_pred_flag = get_bit(gb);
  }
  if (info.slice_type == 0 || info.slice_type == 5 ||  // P
      info.slice_type == 1 || info.slice_type == 6 ||  // B
      info.slice_type == 3 || info.slice_type == 8     // SP
      ) {
    bool num_ref_idx_active_override_flag = get_bit(gb);
    if (num_ref_idx_active_override_flag) {
      info.num_ref_idx_l0_active = get_ue_golomb(gb);
      if (info.slice_type == 1 || info.slice_type == 6) {
        info.num_ref_idx_l1_active = get_ue_golomb(gb);
      }
    } else {
      info.num_ref_idx_l0_active = pps.num_ref_idx_l0_default_active;
      info.num_ref_idx_l1_active = pps.num_ref_idx_l1_default_active;
      ;
    }
  }
  return true;
}

inline bool is_new_access_unit(std::map<u32, SPS>& sps_map,
                               std::map<u32, PPS>& pps_map, SliceHeader& prev,
                               SliceHeader& curr) {
  SPS& prev_sps = sps_map.at(prev.sps_id);
  SPS& curr_sps = sps_map.at(curr.sps_id);
  PPS& curr_pps = pps_map.at(curr.pps_id);
  if (curr.nal_unit_type != 5 && curr.frame_num != prev.frame_num) {
    VLOG(1) << "frame num";
    return true;
  } else if (prev.pps_id != curr.pps_id) {
    VLOG(1) << "pps";
    return true;
  } else if (prev.field_pic_flag != curr.field_pic_flag) {
    VLOG(1) << "field pic";
    return true;
  } else if ((prev.bottom_field_flag != -1 && curr.bottom_field_flag != -1) &&
             prev.bottom_field_flag != curr.bottom_field_flag) {
    VLOG(1) << "bottom field";
    return true;
  } else if ((prev.nal_ref_idc == 0 || curr.nal_ref_idc == 0) &&
             prev.nal_ref_idc != curr.nal_ref_idc) {
    VLOG(1) << "nal ref";
    return true;
  } else if ((prev_sps.poc_type == 0 && curr_sps.poc_type == 0) &&
             (prev.pic_order_cnt_lsb != curr.pic_order_cnt_lsb ||
              prev.delta_pic_order_cnt_bottom !=
                  curr.delta_pic_order_cnt_bottom)) {
    VLOG(1) << "poc type 0: " << prev.pic_order_cnt_lsb << " vs. "
            << curr.pic_order_cnt_lsb << ", " << prev.delta_pic_order_cnt_bottom
            << " vs. " << curr.delta_pic_order_cnt_bottom;
    return true;
  } else if ((prev_sps.poc_type == 1 && curr_sps.poc_type == 1) &&
             (prev.delta_pic_order_cnt[0] != curr.delta_pic_order_cnt[0] ||
              prev.delta_pic_order_cnt[1] != curr.delta_pic_order_cnt[1])) {
    VLOG(1) << "poc type 1";
    return true;
  } else if ((prev.nal_unit_type == 5 || curr.nal_unit_type == 5) &&
             prev.nal_unit_type != curr.nal_unit_type) {
    VLOG(1) << "nal unit type";
    return true;
  } else if ((prev.nal_unit_type == 5 && curr.nal_unit_type == 5) &&
             prev.idr_pic_id != curr.idr_pic_id) {
    VLOG(1) << "idr";
    return true;
  }
  return false;
}
}
