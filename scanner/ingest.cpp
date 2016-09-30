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

#include "scanner/ingest.h"

#include "scanner/util/common.h"
#include "scanner/util/util.h"

#include "storehouse/storage_backend.h"

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
using storehouse::exit_on_error;

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
  i32 video_stream_index;
  AVBitStreamFilterContext* annexb;
};

CodecState setup_video_codec(BufferData* buffer) {
  printf("Setting up video codec\n");
  CodecState state;
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
  printf("Opening input file to read format\n");
  if (avformat_open_input(&state.format_context, NULL, NULL, NULL) < 0) {
    fprintf(stderr, "open input failed\n");
    assert(false);
  }
  // Some formats don't have a header
  if (avformat_find_stream_info(state.format_context, NULL) < 0) {
    fprintf(stderr, "find stream info failed\n");
    assert(false);
  }

  av_dump_format(state.format_context, 0, NULL, 0);

  // Find the best video stream in our input video
  state.video_stream_index = av_find_best_stream(
      state.format_context, AVMEDIA_TYPE_VIDEO, -1 /* auto select */,
      -1 /* no related stream */, &state.in_codec, 0 /* flags */);
  if (state.video_stream_index < 0) {
    fprintf(stderr, "could not find best stream\n");
    assert(false);
  }

  AVStream const* const in_stream =
      state.format_context->streams[state.video_stream_index];

  state.in_codec = avcodec_find_decoder(AV_CODEC_ID_H264);
  if (state.in_codec == NULL) {
    fprintf(stderr, "could not find h264 decoder\n");
    exit(EXIT_FAILURE);
  }

  state.in_cc = avcodec_alloc_context3(state.in_codec);
  if (avcodec_copy_context(state.in_cc, in_stream->codec) < 0) {
    fprintf(stderr, "could not copy codec context from input stream\n");
    exit(EXIT_FAILURE);
  }

  if (avcodec_open2(state.in_cc, state.in_codec, NULL) < 0) {
    fprintf(stderr, "could not open codec\n");
    assert(false);
  }

  state.annexb = av_bitstream_filter_init("h264_mp4toannexb");

  return state;
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

bool read_timestamps(std::string video_path, DatasetItemWebTimestamps& meta) {
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

  CodecState state = setup_video_codec(&buffer);

  AVStream const* const in_stream =
      state.format_context->streams[state.video_stream_index];

  meta.time_base_numerator = in_stream->time_base.num;
  meta.time_base_denominator = in_stream->time_base.den;
  std::vector<i64>& pts_timestamps = meta.pts_timestamps;
  std::vector<i64>& dts_timestamps = meta.dts_timestamps;

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
      fprintf(stderr, "Error while decoding frame %d (%d): %s\n", frame, err,
              err_msg);

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
        fprintf(stderr, "Error while decoding frame %d (%d): %s\n", frame, len,
                err_msg);
        assert(false);
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

  return true;
}

void next_nal(u8*& buffer, i32& buffer_size_left, u8*& nal_start,
              i32& nal_size) {
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

bool preprocess_video(storehouse::StorageBackend* storage,
                      const std::string& dataset_name,
                      const std::string& video_path,
                      const std::string& item_name,
                      DatasetItemMetadata& video_metadata) {
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

  CodecState state = setup_video_codec(&buffer);

  video_metadata.width = state.in_cc->coded_width;
  video_metadata.height = state.in_cc->coded_height;
  video_metadata.chroma_format = VideoChromaFormat::YUV_420;
  video_metadata.codec_type = VideoCodecType::H264;

  std::vector<u8>& metadata_bytes = video_metadata.metadata_packets;
  std::vector<u8> bytestream_bytes;
  std::vector<i64>& keyframe_positions = video_metadata.keyframe_positions;
  std::vector<i64>& keyframe_timestamps = video_metadata.keyframe_timestamps;
  std::vector<i64>& keyframe_byte_offsets =
      video_metadata.keyframe_byte_offsets;

  bool succeeded = true;
  i32 frame = 0;
  bool extradata_extracted = false;
  bool in_meta_packet_sequence = false;
  i64 meta_packet_sequence_start_offset = 0;
  while (true) {
    // Read from format context
    i32 err = av_read_frame(state.format_context, &state.av_packet);
    if (err == AVERROR_EOF) {
      av_packet_unref(&state.av_packet);
      break;
    } else if (err != 0) {
      char err_msg[256];
      av_strerror(err, err_msg, 256);
      fprintf(stderr, "Error while decoding frame %d (%d): %s\n", frame, err,
              err_msg);
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
    av_bitstream_filter_filter(state.annexb, state.in_cc, NULL, &filtered_data,
                               &filtered_data_size, state.av_packet.data,
                               state.av_packet.size,
                               state.av_packet.flags & AV_PKT_FLAG_KEY);

    if (!extradata_extracted) {
      u8* extradata = state.in_cc->extradata;
      i32 extradata_size_left = state.in_cc->extradata_size;

      metadata_bytes.resize(extradata_size_left);
      memcpy(metadata_bytes.data(), extradata, extradata_size_left);

      while (extradata_size_left > 3) {
        u8* nal_start = nullptr;
        i32 nal_size = 0;
        next_nal(extradata, extradata_size_left, nal_start, nal_size);
        i32 nal_ref_idc = (*nal_start >> 5);
        i32 nal_unit_type = (*nal_start) & 0x1F;
        // printf("extradata nal size: %d, nal ref %d, nal unit %d\n",
        //        nal_size, nal_ref_idc, nal_unit_type);
      }
      extradata_extracted = true;
    }

    i64 nal_bytestream_offset = bytestream_bytes.size();
    bytestream_bytes.resize(bytestream_bytes.size() + filtered_data_size +
                            sizeof(i32));
    *((i32*)(bytestream_bytes.data() + nal_bytestream_offset)) =
        filtered_data_size;
    memcpy(bytestream_bytes.data() + nal_bytestream_offset + sizeof(i32),
           filtered_data, filtered_data_size);

    // Parse NAL unit
    u8* nal_parse = filtered_data;
    i32 size_left = filtered_data_size;
    while (size_left > 3) {
      u8* nal_start = nullptr;
      i32 nal_size = 0;
      next_nal(nal_parse, size_left, nal_start, nal_size);

      i32 nal_ref_idc = (*nal_start >> 5);
      i32 nal_unit_type = (*nal_start) & 0x1F;
      // printf("nal size: %d, nal ref %d, nal unit %d\n",
      //        nal_size, nal_ref_idc, nal_unit_type);
      if (nal_unit_type > 4) {
        if (!in_meta_packet_sequence) {
          meta_packet_sequence_start_offset = nal_bytestream_offset;
          in_meta_packet_sequence = true;
        }
      } else {
        in_meta_packet_sequence = false;
      }
      if (nal_unit_type < 6) {
        frame++;
      }
    }

    if (state.av_packet.flags & AV_PKT_FLAG_KEY) {
      // printf("av packet keyframe pts %d\n",
      //        state.av_packet.pts);
      keyframe_byte_offsets.push_back(nal_bytestream_offset);
      keyframe_positions.push_back(frame - 1);
      keyframe_timestamps.push_back(state.av_packet.pts);
      in_meta_packet_sequence = false;
    }

    av_packet_unref(&state.av_packet);
  }

  // Cleanup video decoder
  cleanup_video_codec(state);

  video_metadata.frames = frame;

  const std::vector<u8>& demuxed_video_stream = bytestream_bytes;

  // Write out our metadata video stream
  {
    std::string metadata_path =
        dataset_item_metadata_path(dataset_name, item_name);
    std::unique_ptr<WriteFile> metadata_file;
    exit_on_error(
        make_unique_write_file(storage, metadata_path, metadata_file));

    serialize_dataset_item_metadata(metadata_file.get(), video_metadata);
    metadata_file->save();
  }

  // Write out our demuxed video stream
  {
    std::string data_path = dataset_item_data_path(dataset_name, item_name);
    std::unique_ptr<WriteFile> output_file{};
    exit_on_error(make_unique_write_file(storage, data_path, output_file));

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
    output_file->save();
  }

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
    exit_on_error(make_unique_write_file(storage, web_video_path, output_file));

    const size_t READ_SIZE = 1024 * 1024;
    std::vector<u8> buffer(READ_SIZE);
    while (file) {
      file.read(reinterpret_cast<char*>(buffer.data()), READ_SIZE);
      size_t size_read = file.gcount();

      StoreResult result;
      EXP_BACKOFF(
          output_file->append(size_read, reinterpret_cast<u8*>(buffer.data())),
          result);
      assert(result == StoreResult::Success ||
             result == StoreResult::EndOfFile);
    }

    output_file->save();
  }

  // Get timestamp info for web video
  DatasetItemWebTimestamps timestamps_meta;
  succeeded = read_timestamps(temp_output_path, timestamps_meta);
  if (!succeeded) {
    fprintf(stderr, "Could not get timestamps from web data\n");
    cleanup_video_codec(state);
    return false;
  }

  printf("time base (%d/%d), orig frames %d, dts size %lu\n",
         timestamps_meta.time_base_numerator,
         timestamps_meta.time_base_denominator, frame,
         timestamps_meta.pts_timestamps.size());

  {
    // Write to database storage
    std::string web_video_timestamp_path =
        dataset_item_video_timestamps_path(dataset_name, item_name);
    std::unique_ptr<WriteFile> output_file{};
    exit_on_error(
        make_unique_write_file(storage, web_video_timestamp_path, output_file));

    serialize_dataset_item_web_timestamps(output_file.get(), timestamps_meta);
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
      dataset_name + "_dataset/last_written.bin";

  // File will not exist when first running ingest so check first
  // and return default value if not there
  storehouse::FileInfo info;
  result = storage->get_file_info(last_written_path, info);
  (void)info;
  if (result == StoreResult::FileDoesNotExist) {
    return -1;
  }

  std::unique_ptr<RandomReadFile> file;
  result = make_unique_random_read_file(storage, last_written_path, file);

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
      dataset_name + "_dataset/last_written.bin";
  std::unique_ptr<WriteFile> file;
  make_unique_write_file(storage, last_written_path, file);

  StoreResult result;
  EXP_BACKOFF(
      file->append(sizeof(i32), reinterpret_cast<const u8*>(&file_index)),
      result);
  exit_on_error(result);
}

}  // end anonymous namespace

void ingest(storehouse::StorageConfig* storage_config,
            const std::string& dataset_name,
            const std::string& video_paths_file) {
  i32 rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (!is_master(rank)) return;

  LOG(INFO) << "Creating dataset " << dataset_name << "..." << std::endl;

  // Read in list of video paths and assign unique name to each
  DatasetDescriptor descriptor{};
  std::vector<std::string>& video_paths = descriptor.original_video_paths;
  std::vector<std::string>& item_names = descriptor.item_names;
  {
    i32 video_count = 0;
    std::fstream fs(video_paths_file, std::fstream::in);
    while (fs) {
      std::string path;
      fs >> path;
      if (path.empty()) continue;
      video_paths.push_back(path);
      item_names.push_back(std::to_string(video_count++));
    }
  }

  storehouse::StorageBackend* storage =
      storehouse::StorageBackend::make_from_config(storage_config);

  // Start from the file after the one we last processed succesfully before
  // crashing/exiting
  i32 last_processed_index = read_last_processed_video(storage, dataset_name);

  // Keep track of videos which we can't parse
  i64& total_frames = descriptor.total_frames;
  total_frames = 0;
  std::vector<std::string> bad_paths;
  for (size_t i = last_processed_index + 1; i < video_paths.size(); ++i) {
    const std::string& path = video_paths[i];
    const std::string& item_name = item_names[i];

    LOG(INFO) << "Ingesting video " << path << "..." << std::endl;

    DatasetItemMetadata video_metadata;
    bool valid_video = preprocess_video(storage, dataset_name, path, item_name,
                                        video_metadata);
    if (!valid_video) {
      LOG(WARNING) << "Failed to ingest video " << path << "! "
                   << "Adding to bad paths file "
                   << "(" << BAD_VIDEOS_FILE_PATH << "in current directory)."
                   << std::endl;
      bad_paths.push_back(path);
    } else {
      total_frames += video_metadata.frames;
      // We are summing into the average variables but we will divide
      // by the number of entries at the end
      descriptor.min_frames =
          std::min(descriptor.min_frames, video_metadata.frames);
      descriptor.average_frames += video_metadata.frames;
      descriptor.max_frames =
          std::max(descriptor.max_frames, video_metadata.frames);

      descriptor.min_width =
          std::min(descriptor.min_width, video_metadata.width);
      descriptor.average_width = video_metadata.width;
      descriptor.max_width =
          std::max(descriptor.max_width, video_metadata.width);

      descriptor.min_height =
          std::min(descriptor.min_height, video_metadata.height);
      descriptor.average_height = video_metadata.height;
      descriptor.max_height =
          std::max(descriptor.max_height, video_metadata.height);

      LOG(INFO) << "Finished ingesting video " << path << "." << std::endl;
    }

    // Track the last succesfully processed dataset so we know where
    // to resume if we crash or exit early
    write_last_processed_video(storage, dataset_name, static_cast<i32>(i));
  }
  if (!bad_paths.empty()) {
    std::fstream bad_paths_file(BAD_VIDEOS_FILE_PATH, std::fstream::out);
    for (const std::string& bad_path : bad_paths) {
      bad_paths_file << bad_path << std::endl;
    }
    bad_paths_file.close();
  }

  descriptor.average_frames /= video_paths.size();
  descriptor.average_width /= total_frames;
  descriptor.average_height /= total_frames;
  // Write out dataset descriptor
  {
    const std::string dataset_file_path = dataset_descriptor_path(dataset_name);
    std::unique_ptr<WriteFile> output_file;
    make_unique_write_file(storage, dataset_file_path, output_file);

    serialize_dataset_descriptor(output_file.get(), descriptor);
  }
  // Reset last processed so that we start from scratch next time
  // TODO(apoms): alternatively we could delete the file but apparently
  // that was never designed into the storage interface!
  write_last_processed_video(storage, dataset_name, -1);

  // Add new dataset name to database metadata so we know it exists
  {
    const std::string db_meta_path = database_metadata_path();

    std::unique_ptr<RandomReadFile> meta_in_file;
    make_unique_random_read_file(storage, db_meta_path, meta_in_file);
    u64 pos = 0;
    DatabaseMetadata meta =
        deserialize_database_metadata(meta_in_file.get(), pos);

    i32 dataset_id = meta.next_dataset_id++;
    meta.dataset_names[dataset_id] = dataset_name;
    meta.dataset_job_ids[dataset_id] = {};

    std::unique_ptr<WriteFile> meta_out_file;
    make_unique_write_file(storage, db_meta_path, meta_out_file);
    serialize_database_metadata(meta_out_file.get(), meta);
  }

  LOG(INFO) << "Finished creating dataset " << dataset_name << "." << std::endl;

  delete storage;
}
}
