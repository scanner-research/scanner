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
#include "lightscan/storage/storage_backend.h"
#include "lightscan/util/cuda.h"

#include <cassert>

#include <cuda.h>
#include <nvcuvid.h>

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

// For hardware decode
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda.h"
}

// Stolen from libavformat/movenc.h
#define FF_MOV_FLAG_FASTSTART             (1 <<  7)

namespace lightscan {

pthread_mutex_t av_mutex;

namespace {

class AVFifoBuffer;

typedef struct CuvidContext
{
    CUvideodecoder cudecoder;
    CUvideoparser cuparser;

    AVBufferRef *hwdevice;
    AVBufferRef *hwframe;

    AVBSFContext *bsf;

    AVFifoBuffer *frame_queue;

    int internal_error;

    cudaVideoCodec codec_type;
    cudaVideoChromaFormat chroma_format;
} CuvidContext;

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

// Taken directly from ffmpeg_cuvid.c
typedef struct CUVIDContext {
    AVBufferRef *hw_frames_ctx;
} CUVIDContext;

typedef struct CodecHardwareInfo {
    /* hwaccel options */
    char  *hwaccel_device;
    enum AVPixelFormat hwaccel_output_format;

    /* hwaccel context */
    void  *hwaccel_ctx;
    void (*hwaccel_uninit)(AVCodecContext *s);
    int  (*hwaccel_get_buffer)(AVCodecContext *s, AVFrame *frame, int flags);
    int  (*hwaccel_retrieve_data)(AVCodecContext *s, AVFrame *frame);
    enum AVPixelFormat hwaccel_pix_fmt;
    enum AVPixelFormat hwaccel_retrieved_pix_fmt;
    AVBufferRef *hw_frames_ctx;
} CodecHardwareInfo;

void cuvid_uninit(AVCodecContext *s) {
  CodecHardwareInfo *ist = (CodecHardwareInfo*)s->opaque;
  CUVIDContext *ctx = (CUVIDContext*)ist->hwaccel_ctx;

  ist->hwaccel_uninit        = NULL;
  ist->hwaccel_get_buffer    = NULL;
  ist->hwaccel_retrieve_data = NULL;

  //av_buffer_unref(&ctx->hw_frames_ctx);
  av_buffer_unref(&ist->hw_frames_ctx);

  av_freep(&ist->hwaccel_ctx);
  av_freep(&s->hwaccel_context);

  av_freep(&s->opaque);
}

void cuvid_ctx_free(AVHWDeviceContext *ctx) {
  AVCUDADeviceContext *hwctx = (AVCUDADeviceContext*)ctx->hwctx;
  cuCtxDestroy(hwctx->cuda_ctx);
}

int cuvid_init(AVCodecContext *cc, CUcontext cuda_ctx) {
  CodecHardwareInfo *ist;
  CUVIDContext *ctx = NULL;
  AVBufferRef *hw_device_ctx = NULL;
  AVCUDADeviceContext *device_hwctx;
  AVHWDeviceContext *device_ctx;
  AVHWFramesContext *hwframe_ctx;
  CUdevice device;
  CUcontext dummy;
  CUresult err;
  int ret = 0;

  ist = (CodecHardwareInfo*)cc->opaque;

  if (!ist) {
    ist = (CodecHardwareInfo*)av_mallocz(sizeof(*ist));
    if (!ist) {
      ret = AVERROR(ENOMEM);
      goto error;
    }
    cc->opaque = ist;
  }

  av_log(NULL, AV_LOG_VERBOSE, "Setting up CUVID decoder\n");

  if (ist->hwaccel_ctx) {
    ctx = (CUVIDContext*)ist->hwaccel_ctx;
  } else {
    ctx = (CUVIDContext*)av_mallocz(sizeof(*ctx));
    if (!ctx) {
      ret = AVERROR(ENOMEM);
      goto error;
    }
  }

  if (!hw_device_ctx) {
    hw_device_ctx = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA);
    if (!hw_device_ctx) {
      av_log(NULL, AV_LOG_ERROR, "av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA) failed\n");
      ret = AVERROR(ENOMEM);
      goto error;
    }

    err = cuInit(0);
    if (err != CUDA_SUCCESS) {
      av_log(NULL, AV_LOG_ERROR, "Could not initialize the CUDA driver API\n");
      ret = AVERROR_UNKNOWN;
      goto error;
    }

    // err = cuDeviceGet(&device, gpu_device_id); 
    // if (err != CUDA_SUCCESS) {
    //   av_log(NULL, AV_LOG_ERROR, "Could not get the device number %d\n", 0);
    //   ret = AVERROR_UNKNOWN;
    //   goto error;
    // }

    // err = cuCtxCreate(&cuda_ctx, CU_CTX_SCHED_BLOCKING_SYNC, device);
    // if (err != CUDA_SUCCESS) {
    //   av_log(NULL, AV_LOG_ERROR, "Error creating a CUDA context\n");
    //   ret = AVERROR_UNKNOWN;
    //   goto error;
    // }

    device_ctx = (AVHWDeviceContext*)hw_device_ctx->data;
    device_ctx->free = cuvid_ctx_free;

    device_hwctx = (AVCUDADeviceContext*)device_ctx->hwctx;
    device_hwctx->cuda_ctx = cuda_ctx;

    // err = cuCtxPopCurrent(&dummy);
    // if (err != CUDA_SUCCESS) {
    //   av_log(NULL, AV_LOG_ERROR, "cuCtxPopCurrent failed\n");
    //   ret = AVERROR_UNKNOWN;
    //   goto error;
    // }

    ret = av_hwdevice_ctx_init(hw_device_ctx);
    if (ret < 0) {
      av_log(NULL, AV_LOG_ERROR, "av_hwdevice_ctx_init failed\n");
      goto error;
    }
  } else {
    device_ctx = (AVHWDeviceContext*)hw_device_ctx->data;
    device_hwctx = (AVCUDADeviceContext*)device_ctx->hwctx;
    cuda_ctx = device_hwctx->cuda_ctx;
  }

  if (device_ctx->type != AV_HWDEVICE_TYPE_CUDA) {
    av_log(NULL, AV_LOG_ERROR, "Hardware device context is already initialized for a diffrent hwaccel.\n");
    ret = AVERROR(EINVAL);
    goto error;
  }

  if (!ctx->hw_frames_ctx) {
    ctx->hw_frames_ctx = av_hwframe_ctx_alloc(hw_device_ctx);
    if (!ctx->hw_frames_ctx) {
      av_log(NULL, AV_LOG_ERROR, "av_hwframe_ctx_alloc failed\n");
      ret = AVERROR(ENOMEM);
      goto error;
    }
    cc->hw_frames_ctx = ctx->hw_frames_ctx;
  }

  /* This is a bit hacky, av_hwframe_ctx_init is called by the cuvid decoder
   * once it has probed the neccesary format information. But as filters/nvenc
   * need to know the format/sw_format, set them here so they are happy.
   * This is fine as long as CUVID doesn't add another supported pix_fmt.
   */
  hwframe_ctx = (AVHWFramesContext*)ctx->hw_frames_ctx->data;
  hwframe_ctx->format = AV_PIX_FMT_CUDA;
  hwframe_ctx->sw_format = AV_PIX_FMT_NV12;
  //hwframe_ctx->width     = cc_->coded_width;
  //hwframe_ctx->height    = cc_->coded_height;

  if (!ist->hwaccel_ctx) {
    ist->hwaccel_ctx = ctx;
    ist->hw_frames_ctx = av_buffer_ref(ctx->hw_frames_ctx);

    ist->hwaccel_uninit = cuvid_uninit;

    if (!ist->hw_frames_ctx) {
      av_log(NULL, AV_LOG_ERROR, "av_buffer_ref failed\n");
      ret = AVERROR(ENOMEM);
      goto error;
    }
  }

  return 0;

error:
  av_freep(&ctx);
  return ret;

cancel:
  av_log(NULL, AV_LOG_ERROR,
         "CUVID hwaccel requested, but impossible to achive.\n");
  return AVERROR(EINVAL);
}

// For custom AVIOContext that loads from memory

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
  AVStream* out_stream;
  AVFormatContext* out_format_context;
  AVCodecContext* out_cc;
};

void set_video_stream_settings(AVStream* stream,
                               AVCodecContext* c,
                               AVCodec* codec,
                               int width, int height,
                               AVRational avg_frame_rate,
                               AVRational time_base,
                               int bit_rate,
                               int bit_rate_tolerance,
                               int gop_size,
                               AVRational sample_aspect_ratio)
{
  stream->r_frame_rate.num = avg_frame_rate.num;
  stream->r_frame_rate.den = avg_frame_rate.den;
  stream->time_base = avg_frame_rate;

  c->codec_id = codec->id;
  c->codec_type = AVMEDIA_TYPE_VIDEO;
  c->profile = FF_PROFILE_H264_HIGH;
  c->pix_fmt = AV_PIX_FMT_YUV420P;
  c->width = width;
  c->height = height;
  c->time_base.num = time_base.num;
  c->time_base.den = time_base.den;
  av_opt_set_int(c, "crf", 15, AV_OPT_SEARCH_CHILDREN);
  // c->bit_rate_tolerance = bit_rate_tolerance;
  // c->gop_size = gop_size;

  //c->sample_aspect_ratio = sample_aspect_ratio;
}

CodecState setup_video_codec(BufferData* buffer) {
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
                       0, buffer, &read_packet, NULL, &seek);
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

  state.in_codec = avcodec_find_decoder_by_name("h264_cuvid");
  if (state.in_codec == NULL) {
    fprintf(stderr, "could not find hardware decoder\n");
    exit(EXIT_FAILURE);
  }

  CUcontext cuda_context;
  CUD_CHECK(cuDevicePrimaryCtxRetain(&cuda_context, 0));

  if (cuvid_init(state.in_cc, cuda_context) < 0) {
    fprintf(stderr, "could not init cuvid codec context\n");
    exit(EXIT_FAILURE);
  }

  if (avcodec_open2(state.in_cc, state.in_codec, NULL) < 0) {
    fprintf(stderr, "could not open codec\n");
    assert(false);
  }

  // Setup output codec and stream that we will use to reencode the input video

#ifdef HAVE_X264_ENCODER
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

  state.out_stream =
    avformat_new_stream(state.out_format_context, out_codec);
  if (state.out_stream == NULL) {
    fprintf(stderr, "Could not allocate stream\n");
    exit(1);
  }

  state.out_stream->id = state.out_format_context->nb_streams - 1;
  AVCodecContext* out_cc = state.out_stream->codec;
  state.out_cc = out_cc;
  avcodec_get_context_defaults3(out_cc, out_codec);

  set_video_stream_settings(
    state.out_stream,
    out_cc,
    out_codec,
    state.in_cc->coded_width, state.in_cc->coded_height,
    in_stream->avg_frame_rate,
    in_stream->time_base,
    state.in_cc->bit_rate,
    state.in_cc->bit_rate_tolerance,
    state.in_cc->gop_size,
    in_stream->sample_aspect_ratio);

  if (output_format->flags & AVFMT_GLOBALHEADER)
    state.out_cc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

  state.out_stream->sample_aspect_ratio = in_stream->sample_aspect_ratio;

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

#endif

  return state;
}

void cleanup_video_codec(CodecState state) {
  avformat_free_context(state.out_format_context);
}

}

VideoSeparator::VideoSeparator(
  CUcontext cuda_context,
  AVCodecContext* cc)
  : cuda_context_(cuda_context),
    cc_(cc),
    parser_(nullptr),
    prev_frame_(0),
    is_metadata_(true),
    is_keyframe_(false),
    decode_time_(0)
{
  CUcontext dummy;

  CUD_CHECK(cuCtxPushCurrent(cuda_context_));

  CUVIDPARSERPARAMS cuparseinfo = {};

  cuparseinfo.CodecType = cudaVideoCodec_H264;
  cuparseinfo.ulMaxNumDecodeSurfaces = 20;
  cuparseinfo.ulMaxDisplayDelay = 4;
  cuparseinfo.pUserData = this;
  cuparseinfo.pfnSequenceCallback =
    VideoSeparator::cuvid_handle_video_sequence;
  cuparseinfo.pfnDecodePicture =
    VideoSeparator::cuvid_handle_picture_decode;
  cuparseinfo.pfnDisplayPicture =
    VideoSeparator::cuvid_handle_picture_display;

  CUVIDEOFORMATEX cuparse_ext = {};
  cuparseinfo.pExtVideoInfo = &cuparse_ext;

  CuvidContext *ctx = reinterpret_cast<CuvidContext*>(cc->priv_data);
  if (cc->codec->id == AV_CODEC_ID_H264 || cc->codec->id == AV_CODEC_ID_HEVC) {
    cuparse_ext.format.seqhdr_data_length = ctx->bsf->par_out->extradata_size;
    memcpy(cuparse_ext.raw_seqhdr_data,
           ctx->bsf->par_out->extradata,
           FFMIN(sizeof(cuparse_ext.raw_seqhdr_data), ctx->bsf->par_out->extradata_size));
  } else if (cc->extradata_size > 0) {
    cuparse_ext.format.seqhdr_data_length = cc->extradata_size;
    memcpy(cuparse_ext.raw_seqhdr_data,
           cc->extradata,
           FFMIN(sizeof(cuparse_ext.raw_seqhdr_data), cc->extradata_size));
  }

  CUD_CHECK(cuvidCreateVideoParser(&parser_, &cuparseinfo));

  CUVIDSOURCEDATAPACKET seq_pkt;
  seq_pkt.payload = cuparse_ext.raw_seqhdr_data;
  seq_pkt.payload_size = cuparse_ext.format.seqhdr_data_length;

  if (seq_pkt.payload && seq_pkt.payload_size) {
    CUD_CHECK(cuvidParseVideoData(parser_, &seq_pkt));
  }

  CUD_CHECK(cuCtxPopCurrent(&dummy));
}

VideoSeparator::~VideoSeparator() {
  if (parser_) {
    cuvidDestroyVideoParser(parser_);
  }
}

bool VideoSeparator::decode(AVPacket* packet) {
  CUD_CHECK(cuCtxPushCurrent(cuda_context_));

  AVPacket filter_packet = { 0 };
  AVPacket filtered_packet = { 0 };
  CUdeviceptr mapped_frame = 0;
  int ret = 0, eret = 0;

  CuvidContext *ctx = reinterpret_cast<CuvidContext*>(cc_->priv_data);
  if (ctx->bsf && packet->size) {
    if ((ret = av_packet_ref(&filter_packet, packet)) < 0) {
      return ret;
    }

    if ((ret = av_bsf_send_packet(ctx->bsf, &filter_packet)) < 0) {
      av_packet_unref(&filter_packet);
      return ret;
    }

    if ((ret = av_bsf_receive_packet(ctx->bsf, &filtered_packet)) < 0) {
      return ret;
    }

    packet = &filtered_packet;
  }

  CUVIDSOURCEDATAPACKET cupkt = {};
  cupkt.payload_size = packet->size;
  cupkt.payload = reinterpret_cast<uint8_t*>(packet->data);

  if (packet->size == 0) {
    cupkt.flags |= CUVID_PKT_ENDOFSTREAM;
  }

  CUD_CHECK(cuvidParseVideoData(parser_, &cupkt));

  if (is_metadata_) {
    size_t prev_size = metadata_packets_.size();
    metadata_packets_.resize(prev_size + packet->size + sizeof(int));
    memcpy(metadata_packets_.data() + prev_size,
           &packet->size,
           sizeof(int));
    memcpy(metadata_packets_.data() + prev_size + sizeof(int),
           packet->data,
           packet->size);
  } else {
    size_t prev_size = bitstream_packets_.size();
    bitstream_packets_.resize(prev_size + packet->size + sizeof(int));
    memcpy(bitstream_packets_.data() + prev_size,
           &packet->size,
           sizeof(int));
    memcpy(bitstream_packets_.data() + prev_size + sizeof(int),
           packet->data,
           packet->size);
    if (is_keyframe_) {
      keyframe_positions_.push_back(prev_frame_ - 1);
      keyframe_timestamps_.push_back(packet->pts);
      keyframe_byte_offsets_.push_back(prev_size);
    }
  }

  CUcontext dummy;
  CUD_CHECK(cuCtxPopCurrent(&dummy));

  return false;
}

const std::vector<char>& VideoSeparator::get_metadata_bytes() {
  return metadata_packets_;
}

const std::vector<char>& VideoSeparator::get_bitstream_bytes() {
  return bitstream_packets_;
}

const std::vector<int64_t>& VideoSeparator::get_keyframe_positions() {
  return keyframe_positions_;
}

const std::vector<int64_t>& VideoSeparator::get_keyframe_timestamps() {
  return keyframe_timestamps_;
}

const std::vector<int64_t>& VideoSeparator::get_keyframe_byte_offsets() {
  return keyframe_byte_offsets_;
}


int VideoSeparator::cuvid_handle_video_sequence(
  void *opaque,
  CUVIDEOFORMAT* format)
{
  VideoSeparator& separator =
    *reinterpret_cast<VideoSeparator*>(opaque);

  CUVIDDECODECREATEINFO cuinfo = {};
  cuinfo.CodecType = format->codec;
  cuinfo.ChromaFormat = format->chroma_format;
  cuinfo.OutputFormat = cudaVideoSurfaceFormat_NV12;

  cuinfo.ulWidth = format->coded_width;
  cuinfo.ulHeight = format->coded_height;
  cuinfo.ulTargetWidth = cuinfo.ulWidth;
  cuinfo.ulTargetHeight = cuinfo.ulHeight;

  cuinfo.target_rect.left = 0;
  cuinfo.target_rect.top = 0;
  cuinfo.target_rect.right = cuinfo.ulWidth;
  cuinfo.target_rect.bottom = cuinfo.ulHeight;

  cuinfo.ulNumDecodeSurfaces = 20;
  cuinfo.ulNumOutputSurfaces = 1;
  cuinfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;

  cuinfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;

  separator.decoder_info_ = cuinfo;

  separator.is_metadata_ = false;
}

int VideoSeparator::cuvid_handle_picture_decode(
  void *opaque,
  CUVIDPICPARAMS* picparams)
{
  VideoSeparator& separator = *reinterpret_cast<VideoSeparator*>(opaque);

  if (picparams->intra_pic_flag) {
    separator.is_keyframe_ = true;
  } else {
    separator.is_keyframe_ = false;
  }
  separator.prev_frame_ += 1;
}

int VideoSeparator::cuvid_handle_picture_display(
  void *opaque,
  CUVIDPARSERDISPINFO* dispinfo)
{
  VideoSeparator& separator = *reinterpret_cast<VideoSeparator*>(opaque);
}


//   // avcodec_close(codec_context);
//   // av_free(codec_context);

//   // Cleanup
//   pthread_mutex_lock(&av_mutex);
//   avformat_close_input(&format_context);
//   pthread_mutex_unlock(&av_mutex);
//   av_freep(&io_context->buffer);
//   av_freep(&io_context);

VideoDecoder::VideoDecoder(
  CUcontext cuda_context,
  VideoMetadata metadata,
  std::vector<char> metadata_packets)
  : max_output_frames_(32),
    max_mapped_frames_(8),
    streams_(max_mapped_frames_),
    cuda_context_(cuda_context),
    metadata_(metadata),
    metadata_packets_(metadata_packets),
    parser_(nullptr),
    decoder_(nullptr),
    mapped_frames_(max_mapped_frames_, 0),
    prev_frame_(0),
    new_frame_(false),
    decode_time_(0),
    profiler_(nullptr)
{
  CUcontext dummy;

  CUD_CHECK(cuCtxPushCurrent(cuda_context_));

  for (int i = 0; i < max_mapped_frames_; ++i) {
    cudaStreamCreate(&streams_[i]);
  }

  CUVIDPARSERPARAMS cuparseinfo = {};

  cuparseinfo.CodecType = metadata.codec_type;
  cuparseinfo.ulMaxNumDecodeSurfaces = max_output_frames_;
  cuparseinfo.ulMaxDisplayDelay = 4;
  cuparseinfo.pUserData = this;
  cuparseinfo.pfnSequenceCallback = VideoDecoder::cuvid_handle_video_sequence;
  cuparseinfo.pfnDecodePicture = VideoDecoder::cuvid_handle_picture_decode;
  cuparseinfo.pfnDisplayPicture = VideoDecoder::cuvid_handle_picture_display;

  CUD_CHECK(cuvidCreateVideoParser(&parser_, &cuparseinfo));

  CUVIDDECODECREATEINFO cuinfo = {};
  cuinfo.CodecType = metadata.codec_type;
  cuinfo.ChromaFormat = metadata.chroma_format;
  cuinfo.OutputFormat = cudaVideoSurfaceFormat_NV12;

  cuinfo.ulWidth = metadata.width;
  cuinfo.ulHeight = metadata.height;
  cuinfo.ulTargetWidth = cuinfo.ulWidth;
  cuinfo.ulTargetHeight = cuinfo.ulHeight;

  cuinfo.target_rect.left = 0;
  cuinfo.target_rect.top = 0;
  cuinfo.target_rect.right = cuinfo.ulWidth;
  cuinfo.target_rect.bottom = cuinfo.ulHeight;

  cuinfo.ulNumDecodeSurfaces = max_output_frames_;
  cuinfo.ulNumOutputSurfaces = max_mapped_frames_;
  cuinfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;

  cuinfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;

  CUD_CHECK(cuvidCreateDecoder(&decoder_, &cuinfo));

  CUD_CHECK(cuCtxPopCurrent(&dummy));

  size_t pos = 0;
  while (pos < metadata_packets.size()) {
    int encoded_packet_size =
      *reinterpret_cast<int*>(metadata_packets.data() + pos);
    pos += sizeof(int);
    char* encoded_packet = metadata_packets.data() + pos;
    pos += encoded_packet_size;

    feed(encoded_packet, encoded_packet_size);
  }
}

VideoDecoder::~VideoDecoder() {
  for (int i = 0; i < max_mapped_frames_; ++i) {
    if (mapped_frames_[i] != 0) {
      CUD_CHECK(cuvidUnmapVideoFrame(decoder_, mapped_frames_[i]));
    }
  }

  if (parser_) {
    cuvidDestroyVideoParser(parser_);
  }

  if (decoder_) {
    cuvidDestroyDecoder(decoder_);
  }

  for (int i = 0; i < max_mapped_frames_; ++i) {
    cudaStreamDestroy(streams_[i]);
  }
}

bool VideoDecoder::feed(
  const char* encoded_buffer,
  size_t encoded_size,
  bool discontinuity)
{
  CUD_CHECK(cuCtxPushCurrent(cuda_context_));

  CUVIDSOURCEDATAPACKET cupkt = {};
  cupkt.payload_size = encoded_size;
  cupkt.payload = reinterpret_cast<const uint8_t*>(encoded_buffer);
  if (discontinuity) {
    cupkt.flags |= CUVID_PKT_DISCONTINUITY;
    while (frame_queue_.size() > 0) {
      // Empty queue because we have a new section of frames
      CUVIDPARSERDISPINFO dispinfo;
      frame_queue_.pop(dispinfo);
    }
  }
  if (encoded_size == 0) {
    cupkt.flags |= CUVID_PKT_ENDOFSTREAM;
  }

  CUD_CHECK(cuvidParseVideoData(parser_, &cupkt));

  // Feed metadata packets after EOS to reinit decoder
  if (encoded_size == 0) {
    size_t pos = 0;
    while (pos < metadata_packets_.size()) {
      int encoded_packet_size =
        *reinterpret_cast<int*>(metadata_packets_.data() + pos);
      pos += sizeof(int);
      char* encoded_packet = metadata_packets_.data() + pos;
      pos += encoded_packet_size;

      feed(encoded_packet, encoded_packet_size);
    }
  }

  CUcontext dummy;
  CUD_CHECK(cuCtxPopCurrent(&dummy));

  return frame_queue_.size() > 0;
}


bool VideoDecoder::discard_frame() {
  CUD_CHECK(cuCtxPushCurrent(cuda_context_));

  if (frame_queue_.size() > 0) {
    CUVIDPARSERDISPINFO dispinfo;
    frame_queue_.pop(dispinfo);
  }

  CUcontext dummy;
  CUD_CHECK(cuCtxPopCurrent(&dummy));

  return frame_queue_.size() > 0;
}

bool VideoDecoder::get_frame(
  char* decoded_buffer,
  size_t decoded_size)
{
  CUD_CHECK(cuCtxPushCurrent(cuda_context_));

  if (frame_queue_.size() > 0) {
    CUVIDPARSERDISPINFO dispinfo;
    frame_queue_.pop(dispinfo);

    CUVIDPROCPARAMS params = {};
    params.progressive_frame = dispinfo.progressive_frame;
    params.second_field = 0;
    params.top_field_first = dispinfo.top_field_first;

    int mapped_frame_index = dispinfo.picture_index % max_mapped_frames_;
    if (mapped_frames_[mapped_frame_index] != 0) {
      auto start_unmap = now();
      CU_CHECK(cudaStreamSynchronize(streams_[mapped_frame_index]));
      CUD_CHECK(cuvidUnmapVideoFrame(decoder_,
                                     mapped_frames_[mapped_frame_index]));
      if (profiler_ != nullptr) {
        profiler_->add_interval("unmap_frame", start_unmap, now());
      }
    }
    auto start_map = now();
    unsigned int pitch = 0;
    CUD_CHECK(cuvidMapVideoFrame(decoder_,
                                 dispinfo.picture_index,
                                 &mapped_frames_[mapped_frame_index],
                                 &pitch,
                                 &params));
    // cuvidMapVideoFrame does not wait for convert kernel to finish so sync
    // TODO(apoms): make this an event insertion and have the async 2d memcpy
    //              depend on the event
    CU_CHECK(cudaStreamSynchronize(0));
    if (profiler_ != nullptr) {
      profiler_->add_interval("map_frame", start_map, now());
    }
    CUdeviceptr mapped_frame = mapped_frames_[mapped_frame_index];
    // HACK(apoms): NVIDIA GPU decoder only outputs NV12 format so we rely
    //              on that here to copy the data properly
    for (int i = 0; i < 2; i++) {
      CU_CHECK(cudaMemcpy2DAsync(
                 decoded_buffer + i * metadata_.width * metadata_.height,
                 metadata_.width, // dst pitch
                 (const void*)(
                   mapped_frame + i * pitch * metadata_.height), // src
                 pitch, // src pitch
                 metadata_.width, // width
                 i == 0 ? metadata_.height : metadata_.height / 2, // height
                 cudaMemcpyDeviceToDevice,
                 streams_[mapped_frame_index]));
    }
  }

  CUcontext dummy;
  CUD_CHECK(cuCtxPopCurrent(&dummy));

  return frame_queue_.size() > 0;
}


int VideoDecoder::decoded_frames_buffered() {
  return static_cast<int>(frame_queue_.size());
}

void VideoDecoder::wait_until_frames_copied() {
  for (int i = 0; i < max_mapped_frames_; ++i) {
    CU_CHECK(cudaStreamSynchronize(streams_[i]));
  }
}

void VideoDecoder::set_profiler(Profiler* profiler) {
  profiler_ = profiler;
}

int VideoDecoder::cuvid_handle_video_sequence(
  void *opaque,
  CUVIDEOFORMAT* format)
{
  VideoDecoder& decoder = *reinterpret_cast<VideoDecoder*>(opaque);

}

int VideoDecoder::cuvid_handle_picture_decode(
  void *opaque,
  CUVIDPICPARAMS* picparams)
{
  VideoDecoder& decoder = *reinterpret_cast<VideoDecoder*>(opaque);

  int mapped_frame_index = picparams->CurrPicIdx % decoder.max_mapped_frames_;
  if (decoder.mapped_frames_[mapped_frame_index] != 0) {
    auto start_unmap = now();
    CU_CHECK(cudaStreamSynchronize(decoder.streams_[mapped_frame_index]));
    CUD_CHECK(cuvidUnmapVideoFrame(decoder.decoder_,
                                   decoder.mapped_frames_[mapped_frame_index]));
    if (decoder.profiler_ != nullptr) {
      decoder.profiler_->add_interval("unmap_frame", start_unmap, now());
    }
    decoder.mapped_frames_[mapped_frame_index] = 0;
  }

  CUD_CHECK(cuvidDecodePicture(decoder.decoder_, picparams));
}

int VideoDecoder::cuvid_handle_picture_display(
  void *opaque,
  CUVIDPARSERDISPINFO* dispinfo)
{
  VideoDecoder& decoder = *reinterpret_cast<VideoDecoder*>(opaque);
  decoder.frame_queue_.push(*dispinfo);
  decoder.prev_frame_++;
}

}
