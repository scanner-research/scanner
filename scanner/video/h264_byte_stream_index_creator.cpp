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

#include "scanner/video/h264_byte_stream_index_creator.h"
#include "scanner/util/common.h"
#include "scanner/util/util.h"
#include "scanner/util/storehouse.h"

#include "storehouse/storage_backend.h"
#include "storehouse/storage_config.h"

#include <glog/logging.h>
#include <thread>

// For video
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

#include <cassert>
#include <fstream>

using storehouse::StoreResult;
using storehouse::WriteFile;
using storehouse::RandomReadFile;


namespace scanner {
namespace internal {

H264ByteStreamIndexCreator::H264ByteStreamIndexCreator(WriteFile* b)
  : demuxed_bytestream_(b) {}

bool H264ByteStreamIndexCreator::feed_packet(u8* data, size_t size) {
  u8* orig_data = data;
  i32 orig_size = size;

  i64 nal_bytestream_offset = bytestream_pos_;

  VLOG(2) << "new packet " << nal_bytestream_offset;
  bool insert_sps_nal = false;
  // Parse NAL unit
  const u8* nal_parse = data;
  i32 size_left = size;
  i32 nals_parsed = 0;

  i32 write_size = 0;
  while (size_left > 3) {
    const u8* nal_start = nullptr;
    i32 nal_size = 0;
    next_nal(nal_parse, size_left, nal_start, nal_size);

    if (size_left < 0 || nal_size < 1) {
      continue;
    }

    i32 nal_ref_idc = (*nal_start >> 5);
    i32 nal_unit_type = (*nal_start) & 0x1F;
    VLOG(2) << "frame " << frame_ << ", nal size " << nal_size
            << ", nal_ref_idc " << nal_ref_idc << ", nal unit "
            << nal_unit_type;
    if (nal_ref_idc == 0) {
      num_non_ref_frames_ += 1;
    }
    if (nal_unit_type > 4) {
      if (!in_meta_packet_sequence_) {
        meta_packet_sequence_start_offset_ = nal_bytestream_offset;
        VLOG(2) << "in meta sequence " << nal_bytestream_offset;
        in_meta_packet_sequence_ = true;
        saw_sps_nal_ = false;
      }
    }
    std::vector<u8> rbsp_buffer;
    rbsp_buffer.reserve(64 * 1024);
    u32 consecutive_zeros = 0;
    i32 bytes = nal_size - 1;
    const u8* pb = nal_start + 1;
    while (bytes > 0) {
      /* Copy the byte into the rbsp, unless it
       * is the 0x03 in a 0x000003 */
      if (consecutive_zeros < 2 || *pb != 0x03) {
        rbsp_buffer.push_back(*pb);
      }
      if (*pb == 0) {
        ++consecutive_zeros;
      } else {
        consecutive_zeros = 0;
      }
      ++pb;
      --bytes;
    }

    // We need to track the last SPS NAL because some streams do
    // not insert an SPS every keyframe and we need to insert it
    // ourselves.
    // fprintf(stderr, "nal_size %d, rbsp size %lu\n", nal_size,
    // rbsp_buffer.size());
    const u8* rbsp_start = rbsp_buffer.data();
    i32 rbsp_size = rbsp_buffer.size();

    // SPS
    if (nal_unit_type == 7) {
      saw_sps_nal_ = true;
      i32 offset = 8;
      GetBitsState gb;
      gb.buffer = rbsp_start;
      gb.offset = 0;
      SPS sps;
      if (!parse_sps(gb, sps)) {
        error_message_ = "Failed to parse sps";
        return false;
      }
      i32 sps_id = sps.sps_id;
      sps_map_[sps_id] = sps;
      last_sps_ = sps.sps_id;

      sps_nal_bytes_[sps_id].clear();
      sps_nal_bytes_[sps_id].insert(sps_nal_bytes_[sps_id].end(), nal_start - 3,
                                    nal_start + nal_size + 3);
      VLOG(2) << "Last SPS NAL (" << sps_id << ", " << offset << ")"
              << " seen at frame " << frame_;
    }
    // PPS
    if (nal_unit_type == 8) {
      GetBitsState gb;
      gb.buffer = rbsp_start;
      gb.offset = 0;
      PPS pps;
      if (!parse_pps(gb, pps)) {
        error_message_ = "Failed to parse pps";
        return false;
      }
      pps_map_[pps.pps_id] = pps;
      last_pps_ = pps.pps_id;
      saw_pps_nal_ = true;
      i32 pps_id = pps.pps_id;
      pps_nal_bytes_[pps_id].clear();
      pps_nal_bytes_[pps_id].insert(pps_nal_bytes_[pps_id].end(), nal_start - 3,
                                   nal_start + nal_size + 3);
      VLOG(2) << "PPS id " << pps.pps_id << ", SPS id " << pps.sps_id
              << ", frame " << frame_;
    }
    if (is_vcl_nal(nal_unit_type)) {
      assert(last_pps_ != -1);
      assert(last_sps_ != -1);
      GetBitsState gb;
      gb.buffer = nal_start;
      gb.offset = 8;
      SliceHeader sh;
      if (!parse_slice_header(gb, sps_map_.at(last_sps_), pps_map_,
                              nal_unit_type, nal_ref_idc, sh)) {
        error_message_ = "Failed to parse slice header";
        return false;
      }
      // printf("ref_idx_l0 %d, ref_idx_l1 %d\n",
      // sh.num_ref_idx_l0_active, sh.num_ref_idx_l1_active);
      if (frame_ == 0 || is_new_access_unit(sps_map_, pps_map_, prev_sh_, sh)) {
        frame_++;
        size_t bytestream_offset;
        if (nal_unit_type == 5) {
          // Insert an SPS NAL if we did not see one in the meta packet
          // sequence
          keyframe_byte_offsets_.push_back(nal_bytestream_offset);
          keyframe_positions_.push_back(frame_ - 1);
          // TODO(apoms): Add timestamp info back in
          keyframe_timestamps_.push_back(frame_ - 1);
          saw_sps_nal_ = false;
          VLOG(2) << "keyframe " << frame_ - 1 << ", byte offset "
                  << meta_packet_sequence_start_offset_;

          // Insert metadata
          VLOG(2) << "inserting sps and pss nals";
          i32 size = orig_size;
          for (auto& kv : sps_nal_bytes_) {
            auto& sps_nal = kv.second;
            size += static_cast<i32>(sps_nal.size());
          }
          for (auto& kv : pps_nal_bytes_) {
            auto& pps_nal = kv.second;
            size += static_cast<i32>(pps_nal.size());
          }

          s_write(demuxed_bytestream_, size);
          for (auto& kv : sps_nal_bytes_) {
            auto& sps_nal = kv.second;
            s_write(demuxed_bytestream_, sps_nal.data(), sps_nal.size());
          }
          for (auto& kv : pps_nal_bytes_) {
            auto& pps_nal = kv.second;
            s_write(demuxed_bytestream_, pps_nal.data(), pps_nal.size());
          }
          // Append the packet to the stream
          s_write(demuxed_bytestream_, orig_data, orig_size);

          bytestream_pos_ += sizeof(size) + size;
        } else {
          s_write(demuxed_bytestream_, orig_size);
          // Append the packet to the stream
          s_write(demuxed_bytestream_, orig_data, orig_size);

          bytestream_pos_ += sizeof(orig_size) + orig_size;
        }
      }
      in_meta_packet_sequence_ = false;
      prev_sh_ = sh;
    }
    nals_parsed_++;
  }
  return true;
}


}
}
