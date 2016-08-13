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

#include "lightscan/ingest.h"
#include "lightscan/util/video.h"
#include "lightscan/storage/storage_backend.h"
#include "lightscan/util/cuda.h"

#include <cassert>

#include <cuda.h>
#include <nvcuvid.h>

namespace lightscan {

bool preprocess_video(
  StorageBackend* storage,
  const std::string& dataset_name,
  const std::string& video_path,
  const std::string& output_name)
{
  // The input video we will be preprocessing
  std::unique_ptr<RandomReadFile> input_file{};
  exit_on_error(
    make_unique_random_read_file(storage, video_path, input_file));

  // Load the entire input
  std::vector<char> video_bytes;
  {
    const size_t READ_SIZE = 1024 * 1024;
    size_t pos = 0;
    while (true) {
      video_bytes.resize(video_bytes.size() + READ_SIZE);
      size_t size_read;
      StoreResult result;
      EXP_BACKOFF(
        input_file->read(pos, READ_SIZE, video_bytes.data() + pos, size_read),
        result);
      assert(result == StoreResult::Success ||
             result == StoreResult::EndOfFile);
      pos += size_read;
      if (result == StoreResult::EndOfFile) {
        video_bytes.resize(video_bytes.size() - (READ_SIZE - size_read));
        break;
      }
    }
  }

  // Setup custom buffer for libavcodec so that we can read from memory instead
  // of from a file
  BufferData buffer;
  buffer.ptr = reinterpret_cast<uint8_t*>(video_bytes.data());
  buffer.size = video_bytes.size();
  buffer.orig_ptr = buffer.ptr;
  buffer.initial_size = buffer.size;

  CodecState state = setup_video_codec(&buffer);

  DatasetItemMetadata video_metadata;
  video_metadata.width = state.in_cc->coded_width;
  video_metadata.height = state.in_cc->coded_height;
  video_metadata.chroma_format = cudaVideoChromaFormat_420;
  video_metadata.codec_type = cudaVideoCodec_H264;

  CUcontext cuda_context;
  CUD_CHECK(cuDevicePrimaryCtxRetain(&cuda_context, 0));
  CUD_CHECK(cuCtxPushCurrent(cuda_context));

  VideoSeparator separator(cuda_context, state.in_cc);

  CuvidContext *cuvid_ctx =
    reinterpret_cast<CuvidContext*>(state.in_cc->priv_data);

  bool succeeded = true;
  int frame = 0;
  while (true) {
    // Read from format context
    int err = av_read_frame(state.format_context, &state.av_packet);
    if (err == AVERROR_EOF) {
      av_packet_unref(&state.av_packet);
      break;
    } else if (err != 0) {
      char err_msg[256];
      av_strerror(err, err_msg, 256);
      fprintf(stderr, "Error while decoding frame %d (%d): %s\n",
              frame, err, err_msg);
      CUD_CHECK(cuDevicePrimaryCtxRelease(0));
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
    uint8_t* orig_data = state.av_packet.data;
    int orig_size = state.av_packet.size;
    while (state.av_packet.size > 0) {
      int got_picture = 0;
      char* dec;
      size_t size;
      separator.decode(&state.av_packet);
      int len = avcodec_decode_video2(state.in_cc,
                                      state.picture,
                                      &got_picture,
                                      &state.av_packet);
      if (len < 0) {
        char err_msg[256];
        av_strerror(len, err_msg, 256);
        fprintf(stderr, "Error while decoding frame %d (%d): %s\n",
                frame, len, err_msg);
        assert(false);
      }
      if (got_picture) {
        state.picture->pts = frame;

        if (state.picture->key_frame == 1) {
          printf("keyframe dts %d\n",
                 state.picture->pkt_dts);
        }
        // the picture is allocated by the decoder. no need to free
        frame++;
      }
      // cuvid decoder uses entire packet without setting size or returning len
      state.av_packet.size = 0;
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

  int got_picture;
  do {
    got_picture = 0;
    separator.decode(&state.av_packet);
    int len = avcodec_decode_video2(state.in_cc,
                                    state.picture,
                                    &got_picture,
                                    &state.av_packet);
    (void)len;
    if (got_picture) {
      if (state.picture->key_frame == 1) {
      }
      // the picture is allocated by the decoder. no need to free
      frame++;
    }
  } while (got_picture);

  video_metadata.frames = frame;
  video_metadata.codec_type = cuvid_ctx->codec_type;
  video_metadata.chroma_format = cuvid_ctx->chroma_format;

  video_metadata.metadata_packets = separator.get_metadata_bytes();
  video_metadata.keyframe_positions = separator.get_keyframe_positions();
  video_metadata.keyframe_timestamps = separator.get_keyframe_positions();
  video_metadata.keyframe_byte_offsets = separator.get_keyframe_byte_offsets();

  const std::vector<char>& demuxed_video_stream =
    separator.get_bitstream_bytes();

  // Write out our metadata video stream
  {
    std::string metadata_path =
      dataset_item_metadata_path(dataset_name, item_name);
    std::unique_ptr<WriteFile> metadata_file;
    exit_on_error(
      make_unique_write_file(storage, metadata_path, metadata_file));

    serialize_dataset_item_metadata(metadata_file, video_metadata);
    metadata_file->save();
  }

  // Write out our demuxed video stream
  {
    std::string data_path = dataset_item_data_path(dataset_name, item_name);
    std::unique_ptr<WriteFile> output_file{};
    exit_on_error(
      make_unique_write_file(storage, data_path, output_file));

    const size_t WRITE_SIZE = 16 * 1024;
    char buffer[WRITE_SIZE];
    size_t pos = 0;
    while (pos != demuxed_video_stream.size()) {
      const size_t size_to_write =
        std::min(WRITE_SIZE, demuxed_video_stream.size() - pos);
      StoreResult result;
      EXP_BACKOFF(
        output_file->append(size_to_write, demuxed_video_stream.data() + pos),
        result);
      assert(result == StoreResult::Success ||
             result == StoreResult::EndOfFile);
      pos += size_to_write;
    }
    output_file->save();
  }

  CUD_CHECK(cuDevicePrimaryCtxRelease(0));

  return succeeded;
}

}
