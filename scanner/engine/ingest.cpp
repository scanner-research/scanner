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

#include "scanner/engine/ingest.h"

#include "scanner/util/common.h"
#include "scanner/util/db.h"
#include "scanner/util/h264.h"
#include "scanner/util/util.h"

#include "storehouse/storage_backend.h"

// For image ingest
#include "jpegwrapper/JPEGReader.h"
#include "lodepng/lodepng.h"
#include "bitmap-cpp/bitmap.h"

#include <glog/logging.h>

#include <mpi.h>

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

namespace {

const std::string BAD_VIDEOS_FILE_PATH = "bad_videos.txt";

struct BufferData {
  u8* ptr;
  size_t size;  // size left in the buffer

  u8* orig_ptr;
  size_t initial_size;
};

// For custom AVIOContext that loads from memory
i32 read_packet(void* opaque, u8* buf, i32 buf_size) {
  BufferData* bd = (BufferData*)opaque;
  buf_size = std::min(static_cast<size_t>(buf_size), bd->size);
  /* copy internal buffer data to buf */
  memcpy(buf, bd->ptr, buf_size);
  bd->ptr += buf_size;
  bd->size -= buf_size;
  return buf_size;
}

i64 seek(void* opaque, i64 offset, i32 whence) {
  BufferData* bd = (BufferData*)opaque;
  {
    switch (whence) {
      case SEEK_SET:
        bd->ptr = bd->orig_ptr + offset;
        bd->size = bd->initial_size - offset;
        break;
      case SEEK_CUR:
        bd->ptr += offset;
        bd->size -= offset;
        break;
      case SEEK_END:
        bd->ptr = bd->orig_ptr + bd->initial_size;
        bd->size = 0;
        break;
      case AVSEEK_SIZE:
        return bd->initial_size;
        break;
    }
    return bd->initial_size - bd->size;
  }
}

struct CodecState {
  AVPacket av_packet;
  AVFrame* picture;
  AVFormatContext* format_context;
  AVIOContext* io_context;
  AVCodec* in_codec;
  AVCodecContext* in_cc;
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 34, 0)
  AVCodecParameters* in_cc_params;
#endif
  i32 video_stream_index;
  AVBitStreamFilterContext* annexb;
};

bool setup_video_codec(BufferData* buffer, CodecState& state) {
  LOG(INFO) << "Setting up video codec";
  av_init_packet(&state.av_packet);
  state.picture = av_frame_alloc();
  state.format_context = avformat_alloc_context();

  size_t avio_context_buffer_size = 4096;
  u8* avio_context_buffer =
      static_cast<u8*>(av_malloc(avio_context_buffer_size));
  state.io_context =
      avio_alloc_context(avio_context_buffer, avio_context_buffer_size, 0,
                         buffer, &read_packet, NULL, &seek);
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

  AVStream const* const in_stream =
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

bool read_timestamps(std::string video_path, WebTimestamps& meta) {
  // Load the entire input
  std::vector<u8> video_bytes;
  {
    // Read input from local path
    std::ifstream file{video_path};

    const size_t READ_SIZE = 1024 * 1024;
    while (file) {
      size_t prev_size = video_bytes.size();
      video_bytes.resize(prev_size + READ_SIZE);
      file.read(reinterpret_cast<char*>(video_bytes.data() + prev_size),
                READ_SIZE);
      size_t size_read = file.gcount();
      if (size_read != READ_SIZE) {
        video_bytes.resize(prev_size + size_read);
      }
    }
  }

  // Setup custom buffer for libavcodec so that we can read from memory instead
  // of from a file
  BufferData buffer;
  buffer.ptr = reinterpret_cast<u8*>(video_bytes.data());
  buffer.size = video_bytes.size();
  buffer.orig_ptr = buffer.ptr;
  buffer.initial_size = buffer.size;

  CodecState state;
  if (!setup_video_codec(&buffer, state)) {
    return false;
  }

  AVStream const* const in_stream =
      state.format_context->streams[state.video_stream_index];

  meta.set_time_base_numerator(in_stream->time_base.num);
  meta.set_time_base_denominator(in_stream->time_base.den);
  std::vector<i64> pts_timestamps;
  std::vector<i64> dts_timestamps;

  bool succeeded = true;
  i32 frame = 0;
  while (true) {
    // Read from format context
    i32 err = av_read_frame(state.format_context, &state.av_packet);
    if (err == AVERROR_EOF) {
      av_packet_unref(&state.av_packet);
      break;
    } else if (err != 0) {
      char err_msg[256];
      av_strerror(err, err_msg, 256);
      LOG(ERROR)
        << "Error while decoding frame " << frame
        << " (" << err << "): " << err_msg;

      cleanup_video_codec(state);
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

    u8* orig_data = state.av_packet.data;
    i32 orig_size = state.av_packet.size;
    while (state.av_packet.size > 0) {
      i32 got_picture = 0;
      i32 len = avcodec_decode_video2(state.in_cc, state.picture, &got_picture,
                                      &state.av_packet);
      if (len < 0) {
        char err_msg[256];
        av_strerror(len, err_msg, 256);
        LOG(FATAL)
          << "Error while decoding frame " << frame
          << " (" << len << "): " << err_msg;
      }
      if (got_picture) {
        state.picture->pts = frame;

        pts_timestamps.push_back(state.picture->pkt_pts);
        dts_timestamps.push_back(state.picture->pkt_dts);

        if (state.picture->key_frame == 1) {
        }
        // the picture is allocated by the decoder. no need to free
        frame++;
      }
      state.av_packet.data += len;
      state.av_packet.size -= len;
    }
    state.av_packet.data = orig_data;
    state.av_packet.size = orig_size;

    av_packet_unref(&state.av_packet);
  }

  // /* some codecs, such as MPEG, transmit the I and P frame with a
  //    latency of one frame. You must do the following to have a
  //    chance to get the last frame of the video */
  state.av_packet.data = NULL;
  state.av_packet.size = 0;

  i32 got_picture;
  do {
    got_picture = 0;
    i32 len = avcodec_decode_video2(state.in_cc, state.picture, &got_picture,
                                    &state.av_packet);
    (void)len;
    if (got_picture) {
      pts_timestamps.push_back(state.picture->pkt_pts);
      dts_timestamps.push_back(state.picture->pkt_dts);

      // the picture is allocated by the decoder. no need to free
      frame++;
    }
  } while (got_picture);

  cleanup_video_codec(state);

  for (i64 ts : pts_timestamps) {
    meta.add_pts_timestamps(ts);
  }
  for (i64 ts : dts_timestamps) {
    meta.add_dts_timestamps(ts);
  }

  return true;
}

bool preprocess_video(storehouse::StorageBackend* storage,
                      const std::string& dataset_name,
                      const std::string& video_path,
                      const std::string& item_name,
                      VideoDescriptor& video_descriptor,
                      bool compute_web_metadata) {
  // Load the entire input
  std::vector<u8> video_bytes;
  {
    // Read input from local path
    std::unique_ptr<RandomReadFile> in_file;
    BACKOFF_FAIL(make_unique_random_read_file(storage, video_path, in_file));

    u64 pos = 0;
    video_bytes = read_entire_file(in_file.get(), pos);
  }

  // Setup custom buffer for libavcodec so that we can read from memory instead
  // of from a file
  BufferData buffer;
  buffer.ptr = reinterpret_cast<u8*>(video_bytes.data());
  buffer.size = video_bytes.size();
  buffer.orig_ptr = buffer.ptr;
  buffer.initial_size = buffer.size;

  CodecState state;
  if (!setup_video_codec(&buffer, state)) {
    return false;
  }

  video_descriptor.set_width(state.in_cc->coded_width);
  video_descriptor.set_height(state.in_cc->coded_height);
  video_descriptor.set_chroma_format(VideoDescriptor::YUV_420);
  video_descriptor.set_codec_type(VideoDescriptor::H264);

  std::vector<u8> metadata_bytes;
  std::vector<u8> bytestream_bytes;
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
  std::vector<u8> sps_nal_bytes;
  std::vector<u8> pps_nal_bytes;

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
      LOG(ERROR)
        << "Error while decoding frame " << frame
        << " (" << err << "): " << err_msg;
      cleanup_video_codec(state);
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

    u8* orig_data = state.av_packet.data;
    i32 orig_size = state.av_packet.size;

    u8* filtered_data;
    i32 filtered_data_size;
    if (av_bitstream_filter_filter(
            state.annexb, state.in_cc, NULL, &filtered_data,
            &filtered_data_size, state.av_packet.data, state.av_packet.size,
            state.av_packet.flags & AV_PKT_FLAG_KEY) < 0) {
      char err_msg[256];
      av_strerror(err, err_msg, 256);
      LOG(ERROR)
        << "Error while filtering " << frame
        << " (" << frame << "): " << err_msg;
      cleanup_video_codec(state);
      return false;
    }

    if (!extradata_extracted) {
      const u8* extradata = state.in_cc->extradata;
      i32 extradata_size_left = state.in_cc->extradata_size;

      metadata_bytes.resize(extradata_size_left);
      memcpy(metadata_bytes.data(), extradata, extradata_size_left);

      while (extradata_size_left > 3) {
        const u8* nal_start = nullptr;
        i32 nal_size = 0;
        next_nal(extradata, extradata_size_left, nal_start, nal_size);
        i32 nal_ref_idc = (*nal_start >> 5);
        i32 nal_unit_type = (*nal_start) & 0x1F;
        LOG(INFO)
          << "extradata nal size: " << nal_size
          << ", nal ref " << nal_ref_idc
          << ", nal unit " << nal_unit_type;
      }
      extradata_extracted = true;
    }

    i64 nal_bytestream_offset = bytestream_bytes.size();

    LOG(INFO) << "new packet " << nal_bytestream_offset;
    bool insert_sps_nal = false;
    // Parse NAL unit
    const u8* nal_parse = filtered_data;
    i32 size_left = filtered_data_size;
    while (size_left > 3) {
      const u8* nal_start = nullptr;
      i32 nal_size = 0;
      next_nal(nal_parse, size_left, nal_start, nal_size);

      i32 nal_ref_idc = (*nal_start >> 5);
      i32 nal_unit_type = (*nal_start) & 0x1F;
      LOG(INFO)
        << "frame " << frame
        << ", nal size " << nal_size
        << ", nal_ref_idc " << nal_ref_idc
        << ", nal unit " << nal_unit_type;
      if (nal_unit_type > 4) {
        if (!in_meta_packet_sequence) {
          LOG(INFO) << "in meta sequence " << nal_bytestream_offset;
          meta_packet_sequence_start_offset = nal_bytestream_offset;
          in_meta_packet_sequence = true;
          saw_sps_nal = false;
        }
      } else {
        in_meta_packet_sequence = false;
      }
      // We need to track the last SPS NAL because some streams do
      // not insert an SPS every keyframe and we need to insert it
      // ourselves.
      if (nal_unit_type == 7) {
        saw_sps_nal = true;
        sps_nal_bytes.insert(sps_nal_bytes.end(), nal_start - 3,
                             nal_start + nal_size + 3);
        i32 offset = 32;
        i32 sps_id = parse_exp_golomb(nal_start, nal_size, offset);
        LOG(INFO)
          << "Last SPS NAL (" << sps_id << ", " << offset << ")"
          << " seen at frame " << frame;
      }
      if (nal_unit_type == 8) {
        i32 offset = 8;
        i32 pps_id = parse_exp_golomb(nal_start, nal_size, offset);
        i32 sps_id = parse_exp_golomb(nal_start, nal_size, offset);
        saw_pps_nal = true;
        pps_nal_bytes.insert(pps_nal_bytes.end(), nal_start - 3,
                             nal_start + nal_size + 3);
        LOG(INFO)
          << "PPS id " << pps_id
          << ", SPS id " << sps_id
          << ", frame " << frame;
      }
      if (is_vcl_nal(nal_unit_type)) {
        frame++;
      }
    }
    size_t bytestream_offset;
    if (state.av_packet.flags & AV_PKT_FLAG_KEY) {
      // Insert an SPS NAL if we did not see one in the meta packet sequence
      keyframe_byte_offsets.push_back(nal_bytestream_offset);
      keyframe_positions.push_back(frame - 1);
      keyframe_timestamps.push_back(state.av_packet.pts);
      in_meta_packet_sequence = false;
      saw_sps_nal = false;
      LOG(INFO)
        << "keyframe " << frame - 1
        << ", byte offset " << meta_packet_sequence_start_offset;

      // Insert metadata
      LOG(INFO) << "inserting sps and pss nals";
      size_t prev_size = bytestream_bytes.size();
      i32 size = filtered_data_size + static_cast<i32>(sps_nal_bytes.size()) +
                 static_cast<i32>(pps_nal_bytes.size());
      bytestream_offset =
          prev_size + sizeof(i32) + sps_nal_bytes.size() + pps_nal_bytes.size();
      bytestream_bytes.resize(prev_size + sizeof(i32) + size);
      *((i32*)(bytestream_bytes.data() + prev_size)) = size;
      memcpy(bytestream_bytes.data() + prev_size + sizeof(i32),
             sps_nal_bytes.data(), sps_nal_bytes.size());
      memcpy(bytestream_bytes.data() + prev_size + sizeof(i32) +
                 sps_nal_bytes.size(),
             pps_nal_bytes.data(), pps_nal_bytes.size());
    } else {
      // Append the packet to the stream
      size_t prev_size = bytestream_bytes.size();
      bytestream_offset = prev_size + sizeof(i32);
      bytestream_bytes.resize(prev_size + sizeof(i32) + filtered_data_size);
      *((i32*)(bytestream_bytes.data() + prev_size)) = filtered_data_size;
    }

    memcpy(bytestream_bytes.data() + bytestream_offset, filtered_data,
           filtered_data_size);

    free(filtered_data);

    av_packet_unref(&state.av_packet);
  }

  // Cleanup video decoder
  cleanup_video_codec(state);

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

  const std::vector<u8>& demuxed_video_stream = bytestream_bytes;

  // Write out our metadata video stream
  {
    std::string metadata_path =
        dataset_item_metadata_path(dataset_name, item_name);
    std::unique_ptr<WriteFile> metadata_file;
    BACKOFF_FAIL(make_unique_write_file(storage, metadata_path, metadata_file));

    VideoMetadata m{video_descriptor};
    serialize_video_metadata(metadata_file.get(), m);
    BACKOFF_FAIL(metadata_file->save());
  }

  // Write out our demuxed video stream
  {
    std::string data_path = dataset_item_data_path(dataset_name, item_name);
    std::unique_ptr<WriteFile> output_file{};
    BACKOFF_FAIL(make_unique_write_file(storage, data_path, output_file));

    const size_t WRITE_SIZE = 16 * 1024;
    u8 buffer[WRITE_SIZE];
    size_t pos = 0;
    while (pos != demuxed_video_stream.size()) {
      const size_t size_to_write =
          std::min(WRITE_SIZE, demuxed_video_stream.size() - pos);
      StoreResult result;
      EXP_BACKOFF(output_file->append(size_to_write,
                                      reinterpret_cast<const u8*>(
                                          demuxed_video_stream.data() + pos)),
                  result);
      assert(result == StoreResult::Success ||
             result == StoreResult::EndOfFile);
      pos += size_to_write;
    }
    BACKOFF_FAIL(output_file->save());
  }

  if (compute_web_metadata) {
    // Create temporary file for writing ffmpeg output to
    std::string temp_output_path;
    FILE* fptr;
    temp_file(&fptr, temp_output_path);
    fclose(fptr);
    temp_output_path += ".mp4";

    // Convert to web friendly format
    // vsync 0 needed to never drop or duplicate frames to match fps
    // alternatively we could figure out how to make output fps same as input
    std::string conversion_command =
        "ffmpeg "
        "-i " +
        video_path +
        " "
        "-vsync 0 "
        "-c:v h264 "
        "-strict -2 "
        "-movflags faststart " +
        temp_output_path;

    std::system(conversion_command.c_str());

    // Copy the web friendly data format into database storage
    {
      // Read input from local path
      std::ifstream file{temp_output_path};

      // Write to database storage
      std::string web_video_path =
          dataset_item_video_path(dataset_name, item_name);
      std::unique_ptr<WriteFile> output_file{};
      BACKOFF_FAIL(
          make_unique_write_file(storage, web_video_path, output_file));

      const size_t READ_SIZE = 1024 * 1024;
      std::vector<u8> buffer(READ_SIZE);
      while (file) {
        file.read(reinterpret_cast<char*>(buffer.data()), READ_SIZE);
        size_t size_read = file.gcount();

        StoreResult result;
        EXP_BACKOFF(output_file->append(size_read,
                                        reinterpret_cast<u8*>(buffer.data())),
                    result);
        assert(result == StoreResult::Success ||
               result == StoreResult::EndOfFile);
      }

      BACKOFF_FAIL(output_file->save());
    }

    // Get timestamp info for web video
    WebTimestamps timestamps_meta;
    succeeded = read_timestamps(temp_output_path, timestamps_meta);
    if (!succeeded) {
      LOG(ERROR) << "Could not get timestamps from web data";
      cleanup_video_codec(state);
      return false;
    }

    LOG(INFO)
      << "time base (" << timestamps_meta.time_base_numerator()
      << "/" << timestamps_meta.time_base_denominator() << ")"
      << ", orig frames " << frame
      << ", dts size " << timestamps_meta.pts_timestamps_size();


    {
      // Write to database storage
      std::string web_video_timestamp_path =
          dataset_item_video_timestamps_path(dataset_name, item_name);
      std::unique_ptr<WriteFile> output_file{};
      BACKOFF_FAIL(make_unique_write_file(storage, web_video_timestamp_path,
                                          output_file));

      serialize_web_timestamps(output_file.get(), timestamps_meta);
    }
  }

  return succeeded;
}

/* read_last_processed_video - read from persistent storage the index
 *   of the last succesfully processed video for the given dataset.
 *   Used to recover from failures midway through the ingest process.
 *
 *   @return: index of the last successfully processed video
 */
i32 read_last_processed_video(storehouse::StorageBackend* storage,
                              const std::string& dataset_name) {
  StoreResult result;

  const std::string last_written_path =
      dataset_directory(dataset_name) + "/last_written.bin";

  // File will not exist when first running ingest so check first
  // and return default value if not there
  storehouse::FileInfo info;
  result = storage->get_file_info(last_written_path, info);
  (void)info;
  if (result == StoreResult::FileDoesNotExist) {
    return -1;
  }

  std::unique_ptr<RandomReadFile> file;
  BACKOFF_FAIL(make_unique_random_read_file(storage, last_written_path, file));

  u64 pos = 0;
  size_t size_read;

  i32 last_processed_video;
  EXP_BACKOFF(
      file->read(pos, sizeof(i32), reinterpret_cast<u8*>(&last_processed_video),
                 size_read),
      result);
  assert(result == StoreResult::Success || result == StoreResult::EndOfFile);
  assert(size_read == sizeof(i32));

  return last_processed_video;
}

/* write_last_processed_video - write to persistent storage the index
 *   of the last succesfully processed video for the given dataset.
 *   Used to recover from failures midway through the ingest process.
 *
 */
void write_last_processed_video(storehouse::StorageBackend* storage,
                                const std::string& dataset_name,
                                i32 file_index) {
  const std::string last_written_path =
      dataset_directory(dataset_name) + "/last_written.bin";
  std::unique_ptr<WriteFile> file;
  BACKOFF_FAIL(make_unique_write_file(storage, last_written_path, file));

  BACKOFF_FAIL(
      file->append(sizeof(i32), reinterpret_cast<const u8*>(&file_index)));
}

void ingest_videos(storehouse::StorageBackend* storage,
                   const std::string& dataset_name,
                   std::vector<std::string>& video_paths,
                   bool compute_web_metadata, DatasetDescriptor& descriptor) {
  // Start from the file after the one we last processed succesfully before
  // crashing/exiting

  // TODO(apoms): It is currently invalid to recover from just the last
  //   processed video index because then the total frames and format stats
  //   will be miscalculated. These values should be included in the snapshot.
  // i32 last_processed_index = read_last_processed_video(storage,
  // dataset_name);
  i32 last_processed_index = -1;

  // Keep track of videos which we can't parse
  i64 total_frames{};

  i64 min_frames{};
  i64 average_frames{};
  i64 max_frames{};

  i64 min_width{};
  i64 average_width{};
  i64 max_width{};

  i64 min_height{};
  i64 average_height{};
  i64 max_height{};

  descriptor.set_type(DatasetType_Video);

  DatasetDescriptor::VideoMetadata& metadata = *descriptor.mutable_video_data();
  std::vector<size_t> bad_video_indices;
  std::vector<std::string> item_names;
  for (i32 i = 0; i < video_paths.size(); ++i) {
    item_names.push_back(std::to_string(i));
  }
  for (size_t i = last_processed_index + 1; i < video_paths.size(); ++i) {
    const std::string& path = video_paths[i];
    const std::string& item_name = item_names[i];

    LOG(INFO) << "Ingesting video " << path << "..." << std::endl;

    VideoDescriptor video_descriptor;
    video_descriptor.set_id(i);
    bool valid_video = preprocess_video(storage, dataset_name, path, item_name,
                                        video_descriptor, compute_web_metadata);
    if (!valid_video) {
      LOG(WARNING) << "Failed to ingest video " << path << "! "
                   << "Adding to bad paths file "
                   << "(" << BAD_VIDEOS_FILE_PATH << "in current directory)."
                   << std::endl;
      bad_video_indices.push_back(i);
    } else {
      total_frames += video_descriptor.frames();
      // We are summing into the average variables but we will divide
      // by the number of entries at the end
      min_frames = std::min(min_frames, (i64)video_descriptor.frames());
      average_frames += video_descriptor.frames();
      max_frames = std::max(max_frames, (i64)video_descriptor.frames());

      min_width = std::min(min_width, (i64)video_descriptor.width());
      average_width = video_descriptor.width();
      max_width = std::max(max_width, (i64)video_descriptor.width());

      min_height = std::min(min_height, (i64)video_descriptor.height());
      average_height = video_descriptor.height();
      max_height = std::max(max_height, (i64)video_descriptor.height());

      LOG(INFO) << "Finished ingesting video " << path << "." << std::endl;
    }

    // Track the last succesfully processed dataset so we know where
    // to resume if we crash or exit early
    write_last_processed_video(storage, dataset_name, static_cast<i32>(i));
  }
  if (!bad_video_indices.empty()) {
    std::fstream bad_paths_file(BAD_VIDEOS_FILE_PATH, std::fstream::out);
    for (size_t i : bad_video_indices) {
      const std::string& bad_path = video_paths[i];
      bad_paths_file << bad_path << std::endl;
    }
    bad_paths_file.close();
  }

  metadata.set_total_frames(total_frames);

  metadata.set_min_frames(min_frames);
  metadata.set_average_frames(average_frames);
  metadata.set_max_frames(max_frames);

  metadata.set_min_width(min_width);
  metadata.set_average_width(average_width);
  metadata.set_max_width(max_width);

  metadata.set_min_height(min_height);
  metadata.set_average_height(average_height);
  metadata.set_max_height(max_height);

  if (total_frames > 0) {
    metadata.set_average_width(average_width / total_frames);
    metadata.set_average_height(average_height / total_frames);
  }

  LOG(INFO)
    << "max width " << max_width
    << ",  max_height " << max_height;

  // Remove bad paths
  std::vector<i32> good_video_ids;
  for (size_t i = 0; i < video_paths.size(); ++i) {
    good_video_ids.push_back(i);
  }
  size_t num_bad_videos = bad_video_indices.size();
  for (size_t i = 0; i < num_bad_videos; ++i) {
    size_t bad_index = bad_video_indices[num_bad_videos - 1 - i];
    video_paths.erase(video_paths.begin() + bad_index);
    item_names.erase(item_names.begin() + bad_index);
    good_video_ids.erase(good_video_ids.begin() + bad_index);
  }

  LOG_IF(FATAL, video_paths.size() == 0)
      << "Dataset would be created with zero videos! Exiting";

  for (size_t i = 0; i < video_paths.size(); ++i) {
    const std::string& video_path = video_paths[i];
    const std::string& item_path = item_names[i];
    metadata.add_original_video_paths(video_path);
    metadata.add_video_names(item_path);
    metadata.add_video_ids(good_video_ids[i]);
  }

  // Reset last processed so that we start from scratch next time
  // TODO(apoms): alternatively we could delete the file but apparently
  // that was never designed into the storage interface!
  write_last_processed_video(storage, dataset_name, -1);
}

void ingest_images(storehouse::StorageBackend* storage,
                   const std::string& dataset_name,
                   std::vector<std::string>& image_paths,
                   bool compute_web_metadata, DatasetDescriptor& descriptor) {
  i64 total_images{};

  i64 min_width{};
  i64 average_width{};
  i64 max_width{};

  i64 min_height{};
  i64 average_height{};
  i64 max_height{};

  descriptor.set_type(DatasetType_Image);

  DatasetDescriptor::ImageMetadata& metadata = *descriptor.mutable_image_data();
  std::vector<ImageFormatGroupDescriptor> format_descriptors;
  std::vector<std::unique_ptr<WriteFile>> output_files;
  std::vector<size_t> image_idx_to_format_group;
  std::vector<size_t> bad_image_indices;
  for (size_t i = 0; i < image_paths.size(); ++i) {
    const std::string& path = image_paths[i];

    LOG(INFO) << "Ingesting image " << path << "..." << std::endl;

    // Figure out file type from extension
    ImageEncodingType image_type;
    {
      std::string base_name = basename_s(path);
      std::vector<std::string> parts;
      split(base_name, '.', parts);
      if (parts.size() != 2) {
        LOG(WARNING) << "File " << path << " does not have a valid name. File "
                     << "must only have a single '.' followed by an image "
                     << "extension. Ignoring this file.";
        bad_image_indices.push_back(i);
        image_idx_to_format_group.push_back(0);
        continue;
      }
      std::string extension = parts[1];
      if (!string_to_image_encoding_type(extension, image_type)) {
        LOG(WARNING)
            << "File " << path << " is not an image with a supported "
            << "type. Supported types are: jpeg, png. Ignoring this file";
        bad_image_indices.push_back(i);
        image_idx_to_format_group.push_back(0);
        continue;
      }
    }

    // Read image data into a buffer to inspect for color space, width,
    // and height
    std::vector<u8> image_bytes;
    {
      std::unique_ptr<RandomReadFile> in_file;
      BACKOFF_FAIL(make_unique_random_read_file(storage, path, in_file));
      u64 pos = 0;
      image_bytes = read_entire_file(in_file.get(), pos);
    }

    LOG(INFO) << "path " << path;
    LOG(INFO) << "image size " << image_bytes.size() / 1024;
    i32 image_width;
    i32 image_height;
    ImageColorSpace color_space;
    switch (image_type) {
    case ImageEncodingType::JPEG: {
      JPEGReader reader;
      reader.header_mem(image_bytes.data(), image_bytes.size());
      if (reader.warnings() != "") {
        LOG(WARNING) << "JPEG file " << path
                     << " header could not be parsed: " << reader.warnings()
                     << ". Ignoring.";
        bad_image_indices.push_back(i);
        image_idx_to_format_group.push_back(0);
        continue;
      }
      image_width = reader.width();
      image_height = reader.height();
      switch (reader.colorSpace()) {
      case JPEG::COLOR_GRAYSCALE:
        color_space = ImageColorSpace::Gray;
        break;
      case JPEG::COLOR_RGB:
      case JPEG::COLOR_YCC:
      case JPEG::COLOR_CMYK:
      case JPEG::COLOR_YCCK:
        color_space = ImageColorSpace::RGB;
        break;
      case JPEG::COLOR_UNKNOWN:
        LOG(WARNING) << "JPEG file " << path << " is of unsupported type: "
                     << "COLOR_UNKNOWN. Ignoring.";
        bad_image_indices.push_back(i);
        image_idx_to_format_group.push_back(0);
        continue;
        break;
      }
      break;
    }
    case ImageEncodingType::PNG: {
      unsigned w;
      unsigned h;
      LodePNGState png_state;
      lodepng_state_init(&png_state);
      unsigned error = lodepng_inspect(
        &w, &h, &png_state,
        reinterpret_cast<const unsigned char*>(image_bytes.data()),
        image_bytes.size());
      if (error) {
        LOG(WARNING) << "PNG file " << path << " header could not be parsed: "
                     << lodepng_error_text(error) << ". Ignoring";
        bad_image_indices.push_back(i);
        image_idx_to_format_group.push_back(0);
        continue;
      }
      image_width = w;
      image_height = h;
      switch (png_state.info_png.color.colortype) {
      case LCT_GREY:
        color_space = ImageColorSpace::Gray;
        break;
      case LCT_RGB:
        color_space = ImageColorSpace::RGB;
        break;
      case LCT_RGBA:
        color_space = ImageColorSpace::RGBA;
        break;
      case LCT_PALETTE:
        // NOTE(apoms): We force a paletted file to RGB
        color_space = ImageColorSpace::RGB;
        break;
        ;
      case LCT_GREY_ALPHA: {
        LOG(WARNING) << "PNG file " << path << " is of unsupported type: "
                     << "GREY_ALPHA. Ignoring.";
        bad_image_indices.push_back(i);
        image_idx_to_format_group.push_back(0);
        continue;
      }
      }

      lodepng_state_cleanup(&png_state);
      break;
    }
    case ImageEncodingType::BMP: {
      bitmap::BitmapMetadata metadata;
      bitmap::DecodeResult result =
        bitmap::bitmap_metadata(image_bytes.data(), image_bytes.size(), metadata);
      if (result != bitmap::DecodeResult::Success) {
        LOG(WARNING) << "BMP file " << path << " is invalid.";
        bad_image_indices.push_back(i);
        image_idx_to_format_group.push_back(0);
        continue;
      }
      image_width = metadata.width;
      image_height = metadata.height;
      color_space = metadata.color_space == bitmap::ImageColorSpace::RGB ?
        ImageColorSpace::RGB :
        ImageColorSpace::RGBA;
      break;
    }
    default:
      assert(false);
    }

    // Check if image is of the same type as an existing set of images. If so,
    // write to that file, otherwise create a new format group.
    i32 format_idx = -1;
    for (i32 i = 0; i < metadata.format_groups_size(); ++i) {
      DatasetDescriptor::ImageMetadata::FormatGroup& group =
          *metadata.mutable_format_groups(i);
      if (image_type == group.encoding_type() &&
          color_space == group.color_space() && image_width == group.width() &&
          image_height == group.height()) {
        format_idx = i;
        break;
      }
    }
    if (format_idx == -1) {
      // Create new format group
      format_descriptors.emplace_back();
      ImageFormatGroupDescriptor& desc = format_descriptors.back();
      DatasetDescriptor::ImageMetadata::FormatGroup& group =
          *metadata.add_format_groups();
      group.set_encoding_type(image_type);
      group.set_color_space(color_space);
      group.set_width(image_width);
      group.set_height(image_height);
      group.set_num_images(1);

      desc.set_encoding_type(image_type);
      desc.set_color_space(color_space);
      desc.set_width(image_width);
      desc.set_height(image_height);
      desc.set_num_images(1);

      format_idx = metadata.format_groups_size() - 1;
      desc.set_id(format_idx);
      // Create output file for writing to format group
      WriteFile* file;
      std::string item_path =
          dataset_item_data_path(dataset_name, std::to_string(format_idx));
      BACKOFF_FAIL(storage->make_write_file(item_path, file));
      output_files.emplace_back(file);
    } else {
      // Add to existing format group
      ImageFormatGroupDescriptor& desc = format_descriptors[format_idx];
      DatasetDescriptor::ImageMetadata::FormatGroup& group =
          *metadata.mutable_format_groups(format_idx);
      group.set_num_images(group.num_images() + 1);
      desc.set_num_images(desc.num_images() + 1);
    }
    image_idx_to_format_group.push_back(format_idx);

    // Write out compressed image data
    std::unique_ptr<WriteFile>& output_file = output_files[format_idx];
    i64 image_size = image_bytes.size();
    BACKOFF_FAIL(output_file->append(image_bytes));

    ImageFormatGroupDescriptor& desc = format_descriptors[format_idx];
    desc.add_compressed_sizes(image_size);

    total_images++;

    // We are summing into the average variables but we will divide
    // by the number of entries at the end
    min_width = std::min(min_width, (i64)image_width);
    average_width += image_width;
    max_width = std::max(max_width, (i64)image_width);

    min_height = std::min(min_height, (i64)image_height);
    average_height += image_height;
    max_height = std::max(max_height, (i64)image_height);

    LOG(INFO) << "Finished ingesting image " << path << "." << std::endl;
  }

  // Write out all image paths which failed the ingest process
  if (!bad_image_indices.empty()) {
    LOG(WARNING) << "Writing the path of all ill formatted images which "
                 << "failed to ingest to " << BAD_VIDEOS_FILE_PATH << ".";

    std::fstream bad_paths_file(BAD_VIDEOS_FILE_PATH, std::fstream::out);
    for (size_t i : bad_image_indices) {
      const std::string& bad_path = image_paths[i];
      bad_paths_file << bad_path << std::endl;
    }
    bad_paths_file.close();
  }

  metadata.set_total_images(total_images);

  metadata.set_min_width(min_width);
  metadata.set_average_width(average_width);
  metadata.set_max_width(max_width);

  metadata.set_min_height(min_height);
  metadata.set_average_height(average_height);
  metadata.set_max_height(max_height);

  if (total_images > 0) {
    metadata.set_average_width(average_width / total_images);
    metadata.set_average_height(average_height / total_images);
  }

  LOG(INFO)
    << "max width " << max_width
    << ",  max_height " << max_height;

  for (size_t i = 0; i < image_paths.size(); ++i) {
    const std::string& path = image_paths[i];
    metadata.add_original_image_paths(path);
  }

  // Remove bad paths
  size_t num_bad_images = bad_image_indices.size();
  for (size_t i = 0; i < num_bad_images; ++i) {
    size_t bad_index = bad_image_indices[num_bad_images - 1 - i];
    image_paths.erase(image_paths.begin() + bad_index);
  }

  LOG_IF(FATAL, total_images == 0)
      << "Dataset would be created with zero images! Exiting.";

  for (size_t i = 0; i < image_paths.size(); ++i) {
    const std::string& path = image_paths[i];
    metadata.add_valid_image_paths(path);
  }

  for (size_t i = 0; i < format_descriptors.size(); ++i) {
    // Flush image binary files
    std::unique_ptr<WriteFile>& file = output_files[i];
    BACKOFF_FAIL(file->save());

    // Write out format descriptors for each group
    ImageFormatGroupDescriptor& desc = format_descriptors[i];
    std::string metadata_path =
        dataset_item_metadata_path(dataset_name, std::to_string(i));
    std::unique_ptr<WriteFile> metadata_file;
    BACKOFF_FAIL(make_unique_write_file(storage, metadata_path, metadata_file));

    ImageFormatGroupMetadata m{desc};
    serialize_image_format_group_metadata(metadata_file.get(), m);
    BACKOFF_FAIL(metadata_file->save());
  }
}

}  // end anonymous namespace

void ingest(storehouse::StorageConfig* storage_config, DatasetType dataset_type,
            const std::string& dataset_name, const std::string& paths_file,
            bool compute_web_metadata) {
  i32 rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (!is_master(rank)) return;

  LOG(INFO) << "Creating dataset " << dataset_name << "..." << std::endl;

  // Read in list of paths for each item in file
  std::vector<std::string> paths;
  {
    i32 video_count = 0;
    std::fstream fs(paths_file, std::fstream::in);
    assert(fs.good());
    while (fs) {
      std::string path;
      fs >> path;
      if (path.empty()) continue;
      paths.push_back(path);
    }
  }

  std::unique_ptr<storehouse::StorageBackend> storage{
      storehouse::StorageBackend::make_from_config(storage_config)};

  DatasetDescriptor descriptor{};
  if (dataset_type == DatasetType_Video) {
    ingest_videos(storage.get(), dataset_name, paths, compute_web_metadata,
                  descriptor);
  } else if (dataset_type == DatasetType_Image) {
    ingest_images(storage.get(), dataset_name, paths, compute_web_metadata,
                  descriptor);
  } else {
    assert(false);
  }

  DatabaseMetadata meta{};
  i32 dataset_id;
  {
    const std::string db_meta_path = database_metadata_path();

    std::unique_ptr<RandomReadFile> meta_in_file;
    BACKOFF_FAIL(make_unique_random_read_file(storage.get(), db_meta_path,
                                              meta_in_file));
    u64 pos = 0;
    meta = deserialize_database_metadata(meta_in_file.get(), pos);

    dataset_id = meta.add_dataset(dataset_name);
  }

  descriptor.set_id(dataset_id);
  descriptor.set_name(dataset_name);

  // Write out dataset descriptor
  {
    const std::string dataset_file_path = dataset_descriptor_path(dataset_name);
    std::unique_ptr<WriteFile> output_file;
    BACKOFF_FAIL(
        make_unique_write_file(storage.get(), dataset_file_path, output_file));

    serialize_dataset_descriptor(output_file.get(), descriptor);
    BACKOFF_FAIL(output_file->save());
  }

  // Add new dataset name to database metadata so we know it exists
  {
    const std::string db_meta_path = database_metadata_path();

    std::unique_ptr<WriteFile> meta_out_file;
    BACKOFF_FAIL(
        make_unique_write_file(storage.get(), db_meta_path, meta_out_file));
    serialize_database_metadata(meta_out_file.get(), meta);
    BACKOFF_FAIL(meta_out_file->save());
  }

  LOG(INFO) << "Finished creating dataset " << dataset_name << "." << std::endl;
}
}
