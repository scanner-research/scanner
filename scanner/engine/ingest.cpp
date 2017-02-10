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

#include "scanner/api/database.h"
#include "scanner/engine/db.h"

#include "scanner/util/common.h"
#include "scanner/util/h264.h"
#include "scanner/util/util.h"

#include "storehouse/storage_backend.h"

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
namespace {

const std::string BAD_VIDEOS_FILE_PATH = "bad_videos.txt";

struct FFStorehouseState {
  std::unique_ptr<RandomReadFile> file;
  size_t size; // total file size
  u64 pos;
};

// For custom AVIOContext that loads from memory
i32 read_packet(void *opaque, u8 *buf, i32 buf_size) {
  FFStorehouseState *fs = (FFStorehouseState *)opaque;
  size_t size_read;
  storehouse::StoreResult result;
  EXP_BACKOFF(fs->file->read(fs->pos, buf_size, buf, size_read), result);
  if (result != storehouse::StoreResult::EndOfFile) {
    exit_on_error(result);
  }

  fs->pos += size_read;
  return static_cast<i32>(size_read);
}

i64 seek(void *opaque, i64 offset, i32 whence) {
  FFStorehouseState *fs = (FFStorehouseState *)opaque;
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

struct CodecState {
  AVPacket av_packet;
  AVFrame *picture;
  AVFormatContext *format_context;
  AVIOContext *io_context;
  AVCodec *in_codec;
  AVCodecContext *in_cc;
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 34, 0)
  AVCodecParameters *in_cc_params;
#endif
  i32 video_stream_index;
  AVBitStreamFilterContext *annexb;
};

bool setup_video_codec(FFStorehouseState *fs, CodecState &state) {
  LOG(INFO) << "Setting up video codec";
  av_init_packet(&state.av_packet);
  state.picture = av_frame_alloc();
  state.format_context = avformat_alloc_context();

  size_t avio_context_buffer_size = 4096;
  u8 *avio_context_buffer =
      static_cast<u8 *>(av_malloc(avio_context_buffer_size));
  state.io_context =
      avio_alloc_context(avio_context_buffer, avio_context_buffer_size, 0, fs,
                         &read_packet, NULL, &seek);
  state.format_context->pb = state.io_context;

  // Read file header
  LOG(INFO) << "Opening input file to read format";
  if (avformat_open_input(&state.format_context, NULL, NULL, NULL) < 0) {
    LOG(ERROR) << "open input failed";
    return false;
  }
  // Some formats don't have a header
  if (avformat_find_stream_info(state.format_context, NULL) < 0) {
    LOG(ERROR) << "find stream info failed";
    return false;
  }

  av_dump_format(state.format_context, 0, NULL, 0);

  // Find the best video stream in our input video
  state.video_stream_index = av_find_best_stream(
      state.format_context, AVMEDIA_TYPE_VIDEO, -1 /* auto select */,
      -1 /* no related stream */, &state.in_codec, 0 /* flags */);
  if (state.video_stream_index < 0) {
    LOG(ERROR) << "could not find best stream";
    return false;
  }

  AVStream const *const in_stream =
      state.format_context->streams[state.video_stream_index];

  state.in_codec = avcodec_find_decoder(AV_CODEC_ID_H264);
  if (state.in_codec == NULL) {
    LOG(FATAL) << "could not find h264 decoder";
  }

  state.in_cc = avcodec_alloc_context3(state.in_codec);
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 34, 0)
  state.in_cc_params = avcodec_parameters_alloc();
  if (avcodec_parameters_from_context(state.in_cc_params, in_stream->codec) <
      0) {
    LOG(ERROR) << "could not copy codec params from input stream";
    return false;
  }
  if (avcodec_parameters_to_context(state.in_cc, state.in_cc_params) < 0) {
    LOG(ERROR) << "could not copy codec params to in cc";
    return false;
  }
#else
  if (avcodec_copy_context(state.in_cc, in_stream->codec) < 0) {
    LOG(ERROR) << "could not copy codec params to in cc";
    return false;
  }
#endif

  if (avcodec_open2(state.in_cc, state.in_codec, NULL) < 0) {
    LOG(ERROR) << "could not open codec";
    return false;
  }

  state.annexb = av_bitstream_filter_init("h264_mp4toannexb");

  return true;
}

void cleanup_video_codec(CodecState state) {
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 34, 0)
  avcodec_parameters_free(&state.in_cc_params);
#endif
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(55, 53, 0)
  avcodec_free_context(&state.in_cc);
#else
  avcodec_close(state.in_cc);
  av_freep(&state.in_cc);
#endif
  avformat_close_input(&state.format_context);
  if (state.io_context) {
    av_freep(&state.io_context->buffer);
    av_freep(&state.io_context);
  }
  av_frame_free(&state.picture);
  av_bitstream_filter_close(state.annexb);
}

bool parse_and_write_video(storehouse::StorageBackend *storage,
                           const std::string &table_name,
                           i32 table_id,
                           const std::string &path,
                           std::string& error_message) {
  proto::TableDescriptor table_desc;
  table_desc.set_id(table_id);
  table_desc.set_name(table_name);
  table_desc.set_job_id(-1);

  {
    Column *frame_col = table_desc.add_columns();
    frame_col->set_name("frame");
    frame_col->set_id(0);
    frame_col->set_type(ColumnType::Video);

    Column *frame_info_col = table_desc.add_columns();
    frame_info_col->set_name("frame_info");
    frame_info_col->set_id(1);
    frame_info_col->set_type(ColumnType::Other);
  }

  // Setup custom buffer for libavcodec so that we can read from a storehouse
  // file instead of a posix file
  FFStorehouseState file_state;
  StoreResult result;
  EXP_BACKOFF(make_unique_random_read_file(storage, path, file_state.file), result);
  if (result != StoreResult::Success) {
    error_message = "Can not open video file";
    return false;
  }

  EXP_BACKOFF(file_state.file->get_size(file_state.size), result);
  if (result != StoreResult::Success) {
    error_message = "Can not get file size";
    return false;
  }

  file_state.pos = 0;

  CodecState state;
  if (!setup_video_codec(&file_state, state)) {
    error_message = "Failed to set up video codec";
    return false;
  }

  VideoMetadata video_meta;
  proto::VideoDescriptor &video_descriptor = video_meta.get_descriptor();
  video_descriptor.set_table_id(table_id);
  video_descriptor.set_column_id(0);
  video_descriptor.set_item_id(0);

  video_descriptor.set_width(state.in_cc->width);
  video_descriptor.set_height(state.in_cc->height);
  video_descriptor.set_chroma_format(proto::VideoDescriptor::YUV_420);
  video_descriptor.set_codec_type(proto::VideoDescriptor::H264);

  std::string data_path = table_item_output_path(table_id, 0, 0);
  std::unique_ptr<WriteFile> demuxed_bytestream{};
  BACKOFF_FAIL(make_unique_write_file(storage, data_path, demuxed_bytestream));

  u64 bytestream_pos = 0;
  std::vector<u8> metadata_bytes;
  std::vector<i64> keyframe_positions;
  std::vector<i64> keyframe_timestamps;
  std::vector<i64> keyframe_byte_offsets;

  bool succeeded = true;
  i32 frame = 0;
  bool extradata_extracted = false;
  bool in_meta_packet_sequence = false;
  i64 meta_packet_sequence_start_offset = 0;
  bool saw_sps_nal = false;
  bool saw_pps_nal = false;
  std::map<u32, SPS> sps_map;
  std::map<u32, PPS> pps_map;
  u32 last_sps = -1;
  u32 last_pps = -1;
  std::vector<u8> sps_nal_bytes;
  std::vector<u8> pps_nal_bytes;
  SliceHeader prev_sh;

  i32 num_non_ref_frames = 0;
  i32 avcodec_frame = 0;
  while (true) {
    // Read from format context
    i32 err = av_read_frame(state.format_context, &state.av_packet);
    if (err == AVERROR_EOF) {
      av_packet_unref(&state.av_packet);
      break;
    } else if (err != 0) {
      char err_msg[256];
      av_strerror(err, err_msg, 256);
      LOG(ERROR) << "Error while decoding frame " << frame << " (" << err
                 << "): " << err_msg;
      cleanup_video_codec(state);
      error_message = "Error while decoding frame " + std::to_string(frame) +
                      " (" + std::to_string(err) + "): " + std::string(err_msg);
      return false;
    }

    if (state.av_packet.stream_index != state.video_stream_index) {
      av_packet_unref(&state.av_packet);
      continue;
    }

    /* NOTE1: some codecs are stream based (mpegvideo, mpegaudio)
       and this is the only method to use them because you cannot
       know the compressed data size before analysing it.

       BUT some other codecs (msmpeg4, mpeg4) are inherently frame
       based, so you must call them with all the data for one
       frame exactly. You must also initialize 'width' and
       'height' before initializing them. */

    /* NOTE2: some codecs allow the raw parameters (frame size,
       sample rate) to be changed at any frame. We handle this, so
       you should also take care of it */

    /* here, we use a stream based decoder (mpeg1video), so we
       feed decoder and see if it could decode a frame */

    u8 *orig_data = state.av_packet.data;
    i32 orig_size = state.av_packet.size;

    u8 *filtered_data;
    i32 filtered_data_size;
    if (av_bitstream_filter_filter(
            state.annexb, state.in_cc, NULL, &filtered_data,
            &filtered_data_size, state.av_packet.data, state.av_packet.size,
            state.av_packet.flags & AV_PKT_FLAG_KEY) < 0) {
      char err_msg[256];
      av_strerror(err, err_msg, 256);
      LOG(ERROR) << "Error while filtering " << frame << " (" << frame
                 << "): " << err_msg;
      cleanup_video_codec(state);
      error_message = "Error while filtering frame " + std::to_string(frame) +
                      " (" + std::to_string(err) + "): " + std::string(err_msg);
      return false;
    }

    if (!extradata_extracted) {
      const u8 *extradata = state.in_cc->extradata;
      i32 extradata_size_left = state.in_cc->extradata_size;

      metadata_bytes.resize(extradata_size_left);
      memcpy(metadata_bytes.data(), extradata, extradata_size_left);

      while (extradata_size_left > 3) {
        const u8 *nal_start = nullptr;
        i32 nal_size = 0;
        next_nal(extradata, extradata_size_left, nal_start, nal_size);
        i32 nal_ref_idc = (*nal_start >> 5);
        i32 nal_unit_type = (*nal_start) & 0x1F;
        LOG(INFO) << "extradata nal size: " << nal_size << ", nal ref "
                  << nal_ref_idc << ", nal unit " << nal_unit_type;
      }
      extradata_extracted = true;
    }

    i64 nal_bytestream_offset = bytestream_pos;

    LOG(INFO) << "new packet " << nal_bytestream_offset;
    bool insert_sps_nal = false;
    // Parse NAL unit
    const u8 *nal_parse = filtered_data;
    i32 size_left = filtered_data_size;
    i32 nals_parsed = 0;
    while (size_left > 3) {
      const u8 *nal_start = nullptr;
      i32 nal_size = 0;
      next_nal(nal_parse, size_left, nal_start, nal_size);

      i32 nal_ref_idc = (*nal_start >> 5);
      i32 nal_unit_type = (*nal_start) & 0x1F;
      LOG(INFO) << "frame " << frame << ", nal size " << nal_size
                << ", nal_ref_idc " << nal_ref_idc << ", nal unit "
                << nal_unit_type;
      if (nal_ref_idc == 0) {
        num_non_ref_frames += 1;
      }
      if (nal_unit_type > 4) {
        if (!in_meta_packet_sequence) {
          meta_packet_sequence_start_offset = nal_bytestream_offset;
          filtered_data_size - size_left;
          LOG(INFO) << "in meta sequence " << nal_bytestream_offset;
          in_meta_packet_sequence = true;
          saw_sps_nal = false;
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
      // fprintf(stderr, "nal_size %d, rbsp size %lu\n", nal_size, rbsp_buffer.size());
      const u8* rbsp_start = rbsp_buffer.data();
      i32 rbsp_size = rbsp_buffer.size();

      // SPS
      if (nal_unit_type == 7) {
        saw_sps_nal = true;
        sps_nal_bytes.insert(sps_nal_bytes.end(), nal_start - 3,
                             nal_start + nal_size + 3);
        i32 offset = 8;
        GetBitsState gb;
        gb.buffer = rbsp_start;
        gb.offset = 0;
        SPS sps;
        if (!parse_sps(gb, sps)) {
          return false;
        }
        i32 sps_id = sps.sps_id;
        sps_map[sps_id] = sps;
        last_sps = sps.sps_id;
        LOG(INFO) << "Last SPS NAL (" << sps_id << ", " << offset << ")"
                  << " seen at frame " << frame;
      }
      // PPS
      if (nal_unit_type == 8) {
        GetBitsState gb;
        gb.buffer = rbsp_start;
        gb.offset = 0;
        PPS pps;
        if (!parse_pps(gb, pps)) {
          return false;
        }
        pps_map[pps.pps_id] = pps;
        last_pps = pps.pps_id;
        saw_pps_nal = true;
        pps_nal_bytes.insert(pps_nal_bytes.end(), nal_start - 3,
                             nal_start + nal_size + 3);
        LOG(INFO) << "PPS id " << pps.pps_id << ", SPS id " << pps.sps_id
                  << ", frame " << frame;
      }
      if (is_vcl_nal(nal_unit_type)) {
        assert(last_pps != -1);
        assert(last_sps != -1);
        GetBitsState gb;
        gb.buffer = nal_start;
        gb.offset = 8;
        SliceHeader sh;
        if(!parse_slice_header(gb, sps_map.at(last_sps), pps_map,
                               nal_unit_type, nal_ref_idc, sh)) {
          return false;
        }
        if (frame == 0 || is_new_access_unit(sps_map, pps_map, prev_sh, sh)) {
          frame++;
          size_t bytestream_offset;
          if (state.av_packet.flags & AV_PKT_FLAG_KEY) {
            // Insert an SPS NAL if we did not see one in the meta packet
            // sequence
            keyframe_byte_offsets.push_back(nal_bytestream_offset);
            keyframe_positions.push_back(frame - 1);
            keyframe_timestamps.push_back(state.av_packet.pts);
            saw_sps_nal = false;
            LOG(INFO) << "keyframe " << frame - 1 << ", byte offset "
                      << meta_packet_sequence_start_offset;

            // Insert metadata
            LOG(INFO) << "inserting sps and pss nals";
            i32 size = filtered_data_size +
                       static_cast<i32>(sps_nal_bytes.size()) +
                       static_cast<i32>(pps_nal_bytes.size());

            s_write(demuxed_bytestream.get(), size);
            s_write(demuxed_bytestream.get(), sps_nal_bytes.data(),
                    sps_nal_bytes.size());
            s_write(demuxed_bytestream.get(), pps_nal_bytes.data(),
                    pps_nal_bytes.size());

            bytestream_pos += sizeof(size) + size;
          } else {
            s_write(demuxed_bytestream.get(), filtered_data_size);
            bytestream_pos += sizeof(filtered_data_size) + filtered_data_size;
          }
        }
        in_meta_packet_sequence = false;
        prev_sh = sh;
      }
      nals_parsed++;
    }
    // Append the packet to the stream
    s_write(demuxed_bytestream.get(), filtered_data, filtered_data_size);

    free(filtered_data);

    av_packet_unref(&state.av_packet);
  }

  // Cleanup video decoder
  cleanup_video_codec(state);

  // Save demuxed stream
  BACKOFF_FAIL(demuxed_bytestream->save());

  table_desc.set_num_rows(frame);
  table_desc.set_rows_per_item(frame);
  video_descriptor.set_frames(frame);
  video_descriptor.set_metadata_packets(metadata_bytes.data(),
                                        metadata_bytes.size());

  for (i64 v : keyframe_positions) {
    video_descriptor.add_keyframe_positions(v);
  }
  for (i64 v : keyframe_timestamps) {
    video_descriptor.add_keyframe_timestamps(v);
  }
  for (i64 v : keyframe_byte_offsets) {
    video_descriptor.add_keyframe_byte_offsets(v);
  }

  // Save our metadata for the frame column
  write_video_metadata(storage, video_meta);

  // Save the table descriptor
  write_table_metadata(storage, TableMetadata(table_desc));

  return succeeded;
}

// void ingest_images(storehouse::StorageBackend* storage,
//                    const std::string& table_name,
//                    std::vector<std::string>& image_paths) {
//   i64 total_images{};

//   i64 min_width{};
//   i64 average_width{};
//   i64 max_width{};

//   i64 min_height{};
//   i64 average_height{};
//   i64 max_height{};

//   descriptor.set_type(DatasetType_Image);

//   DatasetDescriptor::ImageMetadata& metadata =
//   *descriptor.mutable_image_data();
//   std::vector<ImageFormatGroupDescriptor> format_descriptors;
//   std::vector<std::unique_ptr<WriteFile>> output_files;
//   std::vector<size_t> image_idx_to_format_group;
//   std::vector<size_t> bad_image_indices;
//   for (size_t i = 0; i < image_paths.size(); ++i) {
//     const std::string& path = image_paths[i];

//     LOG(INFO) << "Ingesting image " << path << "..." << std::endl;

//     // Figure out file type from extension
//     ImageEncodingType image_type;
//     {
//       std::string base_name = basename_s(path);
//       std::vector<std::string> parts;
//       split(base_name, '.', parts);
//       if (parts.size() != 2) {
//         LOG(WARNING) << "File " << path << " does not have a valid name. File
//         "
//                      << "must only have a single '.' followed by an image "
//                      << "extension. Ignoring this file.";
//         bad_image_indices.push_back(i);
//         image_idx_to_format_group.push_back(0);
//         continue;
//       }
//       std::string extension = parts[1];
//       if (!string_to_image_encoding_type(extension, image_type)) {
//         LOG(WARNING)
//             << "File " << path << " is not an image with a supported "
//             << "type. Supported types are: jpeg, png. Ignoring this file";
//         bad_image_indices.push_back(i);
//         image_idx_to_format_group.push_back(0);
//         continue;
//       }
//     }

//     // Read image data into a buffer to inspect for color space, width,
//     // and height
//     std::vector<u8> image_bytes;
//     {
//       std::unique_ptr<RandomReadFile> in_file;
//       BACKOFF_FAIL(make_unique_random_read_file(storage, path, in_file));
//       u64 pos = 0;
//       image_bytes = read_entire_file(in_file.get(), pos);
//     }

//     LOG(INFO) << "path " << path;
//     LOG(INFO) << "image size " << image_bytes.size() / 1024;
//     i32 image_width;
//     i32 image_height;
//     ImageColorSpace color_space;
//     switch (image_type) {
//     case ImageEncodingType::JPEG: {
//       JPEGReader reader;
//       reader.header_mem(image_bytes.data(), image_bytes.size());
//       if (reader.warnings() != "") {
//         LOG(WARNING) << "JPEG file " << path
//                      << " header could not be parsed: " << reader.warnings()
//                      << ". Ignoring.";
//         bad_image_indices.push_back(i);
//         image_idx_to_format_group.push_back(0);
//         continue;
//       }
//       image_width = reader.width();
//       image_height = reader.height();
//       switch (reader.colorSpace()) {
//       case JPEG::COLOR_GRAYSCALE:
//         color_space = ImageColorSpace::Gray;
//         break;
//       case JPEG::COLOR_RGB:
//       case JPEG::COLOR_YCC:
//       case JPEG::COLOR_CMYK:
//       case JPEG::COLOR_YCCK:
//         color_space = ImageColorSpace::RGB;
//         break;
//       case JPEG::COLOR_UNKNOWN:
//         LOG(WARNING) << "JPEG file " << path << " is of unsupported type: "
//                      << "COLOR_UNKNOWN. Ignoring.";
//         bad_image_indices.push_back(i);
//         image_idx_to_format_group.push_back(0);
//         continue;
//         break;
//       }
//       break;
//     }
//     case ImageEncodingType::PNG: {
//       unsigned w;
//       unsigned h;
//       LodePNGState png_state;
//       lodepng_state_init(&png_state);
//       unsigned error = lodepng_inspect(
//         &w, &h, &png_state,
//         reinterpret_cast<const unsigned char*>(image_bytes.data()),
//         image_bytes.size());
//       if (error) {
//         LOG(WARNING) << "PNG file " << path << " header could not be parsed:
//         "
//                      << lodepng_error_text(error) << ". Ignoring";
//         bad_image_indices.push_back(i);
//         image_idx_to_format_group.push_back(0);
//         continue;
//       }
//       image_width = w;
//       image_height = h;
//       switch (png_state.info_png.color.colortype) {
//       case LCT_GREY:
//         color_space = ImageColorSpace::Gray;
//         break;
//       case LCT_RGB:
//         color_space = ImageColorSpace::RGB;
//         break;
//       case LCT_RGBA:
//         color_space = ImageColorSpace::RGBA;
//         break;
//       case LCT_PALETTE:
//         // NOTE(apoms): We force a paletted file to RGB
//         color_space = ImageColorSpace::RGB;
//         break;
//         ;
//       case LCT_GREY_ALPHA: {
//         LOG(WARNING) << "PNG file " << path << " is of unsupported type: "
//                      << "GREY_ALPHA. Ignoring.";
//         bad_image_indices.push_back(i);
//         image_idx_to_format_group.push_back(0);
//         continue;
//       }
//       }

//       lodepng_state_cleanup(&png_state);
//       break;
//     }
//     case ImageEncodingType::BMP: {
//       bitmap::BitmapMetadata metadata;
//       bitmap::DecodeResult result =
//         bitmap::bitmap_metadata(image_bytes.data(), image_bytes.size(),
//         metadata);
//       if (result != bitmap::DecodeResult::Success) {
//         LOG(WARNING) << "BMP file " << path << " is invalid.";
//         bad_image_indices.push_back(i);
//         image_idx_to_format_group.push_back(0);
//         continue;
//       }
//       image_width = metadata.width;
//       image_height = metadata.height;
//       color_space = metadata.color_space == bitmap::ImageColorSpace::RGB ?
//         ImageColorSpace::RGB :
//         ImageColorSpace::RGBA;
//       break;
//     }
//     default:
//       assert(false);
//     }

//     // Check if image is of the same type as an existing set of images. If
//     so,
//     // write to that file, otherwise create a new format group.
//     i32 format_idx = -1;
//     for (i32 i = 0; i < metadata.format_groups_size(); ++i) {
//       DatasetDescriptor::ImageMetadata::FormatGroup& group =
//           *metadata.mutable_format_groups(i);
//       if (image_type == group.encoding_type() &&
//           color_space == group.color_space() && image_width == group.width()
//           &&
//           image_height == group.height()) {
//         format_idx = i;
//         break;
//       }
//     }
//     if (format_idx == -1) {
//       // Create new format group
//       format_descriptors.emplace_back();
//       ImageFormatGroupDescriptor& desc = format_descriptors.back();
//       DatasetDescriptor::ImageMetadata::FormatGroup& group =
//           *metadata.add_format_groups();
//       group.set_encoding_type(image_type);
//       group.set_color_space(color_space);
//       group.set_width(image_width);
//       group.set_height(image_height);
//       group.set_num_images(1);

//       desc.set_encoding_type(image_type);
//       desc.set_color_space(color_space);
//       desc.set_width(image_width);
//       desc.set_height(image_height);
//       desc.set_num_images(1);

//       format_idx = metadata.format_groups_size() - 1;
//       desc.set_id(format_idx);
//       // Create output file for writing to format group
//       WriteFile* file;
//       std::string item_path =
//           dataset_item_data_path(dataset_name, std::to_string(format_idx));
//       BACKOFF_FAIL(storage->make_write_file(item_path, file));
//       output_files.emplace_back(file);
//     } else {
//       // Add to existing format group
//       ImageFormatGroupDescriptor& desc = format_descriptors[format_idx];
//       DatasetDescriptor::ImageMetadata::FormatGroup& group =
//           *metadata.mutable_format_groups(format_idx);
//       group.set_num_images(group.num_images() + 1);
//       desc.set_num_images(desc.num_images() + 1);
//     }
//     image_idx_to_format_group.push_back(format_idx);

//     // Write out compressed image data
//     std::unique_ptr<WriteFile>& output_file = output_files[format_idx];
//     i64 image_size = image_bytes.size();
//     BACKOFF_FAIL(output_file->append(image_bytes));

//     ImageFormatGroupDescriptor& desc = format_descriptors[format_idx];
//     desc.add_compressed_sizes(image_size);

//     total_images++;

//     // We are summing into the average variables but we will divide
//     // by the number of entries at the end
//     min_width = std::min(min_width, (i64)image_width);
//     average_width += image_width;
//     max_width = std::max(max_width, (i64)image_width);

//     min_height = std::min(min_height, (i64)image_height);
//     average_height += image_height;
//     max_height = std::max(max_height, (i64)image_height);

//     LOG(INFO) << "Finished ingesting image " << path << "." << std::endl;
//   }

//   // Write out all image paths which failed the ingest process
//   if (!bad_image_indices.empty()) {
//     LOG(WARNING) << "Writing the path of all ill formatted images which "
//                  << "failed to ingest to " << BAD_VIDEOS_FILE_PATH << ".";

//     std::fstream bad_paths_file(BAD_VIDEOS_FILE_PATH, std::fstream::out);
//     for (size_t i : bad_image_indices) {
//       const std::string& bad_path = image_paths[i];
//       bad_paths_file << bad_path << std::endl;
//     }
//     bad_paths_file.close();
//   }

//   metadata.set_total_images(total_images);

//   metadata.set_min_width(min_width);
//   metadata.set_average_width(average_width);
//   metadata.set_max_width(max_width);

//   metadata.set_min_height(min_height);
//   metadata.set_average_height(average_height);
//   metadata.set_max_height(max_height);

//   if (total_images > 0) {
//     metadata.set_average_width(average_width / total_images);
//     metadata.set_average_height(average_height / total_images);
//   }

//   LOG(INFO)
//     << "max width " << max_width
//     << ",  max_height " << max_height;

//   for (size_t i = 0; i < image_paths.size(); ++i) {
//     const std::string& path = image_paths[i];
//     metadata.add_original_image_paths(path);
//   }

//   // Remove bad paths
//   size_t num_bad_images = bad_image_indices.size();
//   for (size_t i = 0; i < num_bad_images; ++i) {
//     size_t bad_index = bad_image_indices[num_bad_images - 1 - i];
//     image_paths.erase(image_paths.begin() + bad_index);
//   }

//   LOG_IF(FATAL, total_images == 0)
//       << "Dataset would be created with zero images! Exiting.";

//   for (size_t i = 0; i < image_paths.size(); ++i) {
//     const std::string& path = image_paths[i];
//     metadata.add_valid_image_paths(path);
//   }

//   for (size_t i = 0; i < format_descriptors.size(); ++i) {
//     // Flush image binary files
//     std::unique_ptr<WriteFile>& file = output_files[i];
//     BACKOFF_FAIL(file->save());

//     // Write out format descriptors for each group
//     ImageFormatGroupDescriptor& desc = format_descriptors[i];
//     std::string metadata_path =
//         dataset_item_metadata_path(dataset_name, std::to_string(i));
//     std::unique_ptr<WriteFile> metadata_file;
//     BACKOFF_FAIL(make_unique_write_file(storage, metadata_path,
//     metadata_file));

//     ImageFormatGroupMetadata m{desc};
//     serialize_image_format_group_metadata(metadata_file.get(), m);
//     BACKOFF_FAIL(metadata_file->save());
//   }
// }
} // end anonymous namespace

Result ingest_videos(storehouse::StorageConfig *storage_config,
                   const std::string &db_path,
                   const std::vector<std::string> &table_names,
                   const std::vector<std::string> &paths,
                   std::vector<FailedVideo> &failed_videos) {
  Result result;
  result.set_success(true);

  internal::set_database_path(db_path);
  av_register_all();

  std::unique_ptr<storehouse::StorageBackend> storage{
      storehouse::StorageBackend::make_from_config(storage_config)};

  internal::DatabaseMetadata meta = internal::read_database_metadata(
      storage.get(), internal::DatabaseMetadata::descriptor_path());
  std::vector<i32> table_ids;
  for (size_t i = 0; i < table_names.size(); ++i) {
    table_ids.push_back(meta.add_table(table_names[i]));
  }
  std::vector<bool> bad_videos(table_names.size(), false);
  std::vector<std::string> bad_messages(table_names.size());
  std::vector<std::thread> ingest_threads;
  i32 num_threads = std::thread::hardware_concurrency();
  i32 videos_allocated = 0;
  for (i32 t = 0; t < num_threads; ++t) {
    i32 to_allocate =
        (table_names.size() - videos_allocated) / (num_threads - t);
    i32 start = videos_allocated;
    videos_allocated += to_allocate;
    videos_allocated = std::min((size_t)videos_allocated, table_names.size());
    i32 end = videos_allocated;
    ingest_threads.emplace_back([&, start, end]() {
      for (i32 i = start; i < end; ++i) {
        printf("%s\n", table_names[i].c_str());
        if (!internal::parse_and_write_video(storage.get(), table_names[i],
                                             table_ids[i], paths[i],
                                             bad_messages[i])) {
          // Did not ingest correctly, skip it
          bad_videos[i] = true;
        }
      }
    });
  }
  for (i32 t = 0; t < num_threads; ++t) {
    ingest_threads[t].join();
  }

  size_t num_bad_videos = 0;
  for (size_t i = 0; i < table_names.size(); ++i) {
    if (bad_videos[i]) {
      num_bad_videos++;
      LOG(WARNING) << "Failed to ingest video " << paths[i] << "!";
      failed_videos.push_back({paths[i], bad_messages[i]});
      meta.remove_table(table_ids[i]);
    }
  }
  if (num_bad_videos == table_names.size()) {
    result.set_success(false);
    result.set_msg("All videos failed to ingest properly");
  }

  // Save the db metadata
  internal::write_database_metadata(storage.get(), meta);

  return result;
}

void ingest_images(storehouse::StorageConfig *storage_config,
                   const std::string &db_path, const std::string &table_name,
                   const std::vector<std::string> &paths) {
  internal::set_database_path(db_path);

  std::unique_ptr<storehouse::StorageBackend> storage{
      storehouse::StorageBackend::make_from_config(storage_config)};

  LOG(FATAL) << "Image ingest under construction!" << std::endl;

  LOG(INFO) << "Creating image table " << table_name << "..." << std::endl;
}
}
}
