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

#include "lightscan/util/video.h"

#include <cassert>

// For video
extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavformat/avio.h"
#include "libavformat/movenc.h"
#include "libavutil/error.h"
#include "libswscale/swscale.h"


#include "libavcodec/avcodec.h"
#include "libavfilter/avfilter.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
#include "libavutil/pixdesc.h"
#include "libavutil/opt.h"
  
}

// Stolen from libavformat/movenc.h
#define FF_MOV_FLAG_FASTSTART             (1 <<  7)

namespace lightscan {

namespace {

struct BufferData {
  uint8_t *ptr;
  size_t size; // size left in the buffer

  uint8_t *orig_ptr;
  size_t initial_size;
};

// For custom AVIOContext that loads from memory

int read_packet(void *opaque, uint8_t *buf, int buf_size) {
  BufferData* bd = (BufferData*)opaque;
  buf_size = std::min(static_cast<size_t>(buf_size), bd->size);
  /* copy internal buffer data to buf */
  memcpy(buf, bd->ptr, buf_size);
  bd->ptr  += buf_size;
  bd->size -= buf_size;
  return buf_size;
}

int64_t seek(void *opaque, int64_t offset, int whence) {
  BufferData* bd = (BufferData*)opaque;
  {
    switch (whence)
    {
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
  // Input objects
  AVFormatContext* format_context;
  AVIOContext* io_context;
  AVCodec* in_codec;
  AVCodecContext* in_cc;
  int video_stream_index;
  // Output objets
  AVFormatContext* out_format_context;
  AVCodecContext* out_cc;
};

}

void set_encoder_context_settings(AVCodecContext* c,
                                  AVCodec* codec,
                                  int width, int height,
                                  AVRational time_base,
                                  int bit_rate,
                                  int bit_rate_tolerance,
                                  int gop_size,
                                  AVRational sample_aspect_ratio)
{
  c->codec_id = codec->id;
  c->codec_type = AVMEDIA_TYPE_VIDEO;
  c->width = width;
  c->height = height;
  c->time_base.den = time_base.den;
  c->time_base.num = time_base.num;
  c->bit_rate = bit_rate;
  c->bit_rate_tolerance = bit_rate_tolerance;
  c->gop_size = gop_size;
  c->pix_fmt = AV_PIX_FMT_YUV420P;
  c->profile = FF_PROFILE_H264_BASELINE;
  c->me_range = 0; // Motion estimation range
  c->max_b_frames = 0;
  c->rc_max_rate = 400000;

  //
  c->qblur = 0.5;
  c->qcompress = 0.5;
  c->b_quant_offset = 1.25;
  c->b_quant_factor = 1.25;
  c->i_quant_offset = 0.0;
  c->i_quant_factor = -0.71;

  c->sample_aspect_ratio = sample_aspect_ratio;
}

CodecState setup_video_codec(BufferData buffer) {
  printf("Setting up video codec\n");
  CodecState state;
  av_init_packet(&state.av_packet);
  state.picture = av_frame_alloc();
  state.format_context = avformat_alloc_context();

  size_t avio_context_buffer_size = 4096;
  uint8_t* avio_context_buffer =
    static_cast<uint8_t*>(av_malloc(avio_context_buffer_size));
  state.io_context =
    avio_alloc_context(avio_context_buffer, avio_context_buffer_size,
                       0, &buffer, &read_packet, NULL, &seek);
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
  state.video_stream_index =
    av_find_best_stream(state.format_context,
                        AVMEDIA_TYPE_VIDEO,
                        -1 /* auto select */,
                        -1 /* no related stream */,
                        &state.in_codec,
                        0 /* flags */);
  if (state.video_stream_index < 0) {
    fprintf(stderr, "could not find best stream\n");
    assert(false);
  }

  AVStream const* const in_stream =
    state.format_context->streams[state.video_stream_index];

  state.in_cc = in_stream->codec;

  if (avcodec_open2(state.in_cc, state.in_codec, NULL) < 0) {
    fprintf(stderr, "could not open codec\n");
    assert(false);
  }

  // Setup output codec and stream that we will use to reencode the input video
  AVOutputFormat* output_format = av_guess_format("mp4", NULL, NULL);
  if (output_format == NULL) {
    fprintf(stderr, "output format could not be guessed\n");
    assert(false);
  }
  avformat_alloc_output_context2(&state.out_format_context,
                                 output_format,
                                 NULL,
                                 NULL);
  printf("output format\n");
  fflush(stdout);

  AVCodec* out_codec = avcodec_find_encoder_by_name("libx264");
  if (out_codec == NULL) {
    fprintf(stderr, "could not find encoder for codec name\n");
    assert(false);
  }

  AVStream* out_stream =
    avformat_new_stream(state.out_format_context, out_codec);
  if (out_stream == NULL) {
    fprintf(stderr, "Could not allocate stream\n");
    exit(1);
  }

  out_stream->id = state.out_format_context->nb_streams - 1;
  AVCodecContext* out_cc = out_stream->codec;
  state.out_cc = out_cc;
  avcodec_get_context_defaults3(out_cc, out_codec);

  printf("width %d\n", state.in_cc->coded_width);
  printf("height %d\n", state.in_cc->coded_height);
  printf("bit_rate %d\n", state.in_cc->bit_rate);
  printf("bit_rate_tolerance %d\n", state.in_cc->bit_rate_tolerance);
  printf("gop_size %d\n", state.in_cc->gop_size);
  printf("me_range %d\n", state.in_cc->me_range);
  printf("b_frame_strategy %d\n", state.in_cc->b_frame_strategy);
  printf("max_b_frames %d\n", state.in_cc->max_b_frames);
  printf("me_range %d\n", state.in_cc->me_range);
  printf("before open codec\n");
  fflush(stdout);

  set_encoder_context_settings(
    out_cc,
    out_codec,
    state.in_cc->coded_width, state.in_cc->coded_height,
    in_stream->time_base,
    state.in_cc->bit_rate,
    state.in_cc->bit_rate_tolerance,
    state.in_cc->gop_size,
    in_stream->sample_aspect_ratio);

  if (output_format->flags & AVFMT_GLOBALHEADER)
    state.out_cc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

  out_stream->sample_aspect_ratio = in_stream->sample_aspect_ratio;

  if (avcodec_open2(out_cc, out_codec, NULL) < 0) {
    fprintf(stderr, "Could not open video codec\n");
    exit(1);
  }

  // if (codec->capabilities & CODEC_CAP_TRUNCATED) {
  //   codec_context->flags |= CODEC_FLAG_TRUNCATED;
  // }

/* For some codecs, such as msmpeg4 and mpeg4, width and height
   MUST be initialized there because this information is not
   available in the bitstream. */

/* the codec gives us the frame size, in samples */
  // codec_context->width = width;
  // codec_context->height = height;
  // codec_context->pix_fmt = AV_AV_PIX_FMT_YUV420P;
  // codec_context->bit_rate = format_;

  // Set fast start to move mov to the end
  MOVMuxContext *mov = NULL;

  mov = (MOVMuxContext *)state.out_format_context->priv_data;
  mov->flags |= FF_MOV_FLAG_FASTSTART;

  return state;
}

void cleanup_video_codec(CodecState state) {
  avformat_free_context(state.out_format_context);
}

void preprocess_video(
  StorageBackend* storage,
  const std::string& video_path,
  const std::string& processed_video_path,
  const std::string& iframe_path)
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

  CodecState state = setup_video_codec(buffer);

  FILE* fp;
  std::string filename;
  temp_file(&fp, filename);

  avio_open2(&state.out_format_context->pb, filename.c_str(), AVIO_FLAG_WRITE,
             NULL, NULL);
  avformat_write_header(state.out_format_context, NULL);

  strcpy(state.out_format_context->filename, filename.c_str());

  AVPacket out_packet = {0};
  av_init_packet(&out_packet);
  int got_out_packet = 0;

  std::vector<int> i_frame_positions;
  int frame = 0;
  while (true) {
    // Read from format context
    int err = av_read_frame(state.format_context, &state.av_packet);
    if (err == AVERROR_EOF) {
      av_packet_unref(&state.av_packet);
      break;
    } else if (err != 0) {
      printf("err %d\n", err);
      assert(err == 0);
    }

    if (state.av_packet.stream_index != state.video_stream_index) {
      av_packet_unref(&state.av_packet);
      continue;
    }

    av_interleaved_write_frame(state.out_format_context, &state.av_packet);

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
    while (state.av_packet.size > 0) {
      int got_picture = 0;
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
        int ret = avcodec_encode_video2(state.out_cc,
                                        &out_packet,
                                        state.picture,
                                        &got_out_packet);
        if (ret < 0) {
          fprintf(stderr, "Error encoding video frame: %s\n", av_err2str(ret));
          exit(1);
        }

        if (got_out_packet) {
          av_interleaved_write_frame(state.out_format_context, &out_packet);
        }

        if (state.picture->key_frame == 1) {
          i_frame_positions.push_back(frame);
        }
        // the picture is allocated by the decoder. no need to free
        frame++;
      }
      state.av_packet.size -= len;
      state.av_packet.data += len;
    }
    state.av_packet.data = orig_data;
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
    int len = avcodec_decode_video2(state.in_cc,
                                    state.picture,
                                    &got_picture,
                                    &state.av_packet);
    (void)len;
    if (got_picture) {
      if (state.picture->key_frame == 1) {
        i_frame_positions.push_back(frame);
      }
      // the picture is allocated by the decoder. no need to free
      frame++;
    }
  } while (got_picture);

  av_write_trailer(state.out_format_context);

  if (!(state.out_format_context->oformat->flags & AVFMT_NOFILE) &&
      state.out_format_context->pb)
    avio_close(state.out_format_context->pb);

  fclose(fp);

  printf("pix_fmt %d\n", state.in_cc->pix_fmt);

  {
    // We will process the input video and write it to this output file
    std::unique_ptr<WriteFile> output_file{};
    exit_on_error(
      make_unique_write_file(storage, processed_video_path, output_file));

    printf("filename %s\n", filename.c_str());
    FILE* read_fp = fopen(filename.c_str(), "r");
    assert(read_fp != nullptr);
    const size_t WRITE_SIZE = 16 * 1024;
    char buffer[WRITE_SIZE];
    size_t pos = 0;
    while (true) {
      size_t size_read = fread(buffer, WRITE_SIZE, 1, read_fp);
      printf("size_read %lu\n", size_read);
      StoreResult result;
      EXP_BACKOFF(
        output_file->append(size_read, buffer), result);
      assert(result == StoreResult::Success ||
             result == StoreResult::EndOfFile);
      pos += size_read;
      if (size_read != 0) {
        break;
      }
    }
    output_file->save();
    fclose(read_fp);
  }


//   meta->num_frames = frame;

//   // avcodec_close(codec_context);
//   // av_free(codec_context);

//   // Cleanup
//   pthread_mutex_lock(&av_mutex);
//   avformat_close_input(&format_context);
//   pthread_mutex_unlock(&av_mutex);
//   av_freep(&io_context->buffer);
//   av_freep(&io_context);

}

}
