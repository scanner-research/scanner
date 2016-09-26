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

#include "scanner/video/video_separator.h"
#include "scanner/util/cuda.h"

#include "storehouse/storage_backend.h"

#include <cassert>

#include <cuda.h>
#include <nvcuvid.h>

// For video
extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavformat/avio.h"
#include "libavutil/error.h"
#include "libswscale/swscale.h"

#include "libavcodec/avcodec.h"
#include "libavfilter/avfilter.h"
#include "libavformat/avformat.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libswscale/swscale.h"

// For hardware decode
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda.h"
}

// Stolen from libavformat/movenc.h
#define FF_MOV_FLAG_FASTSTART (1 << 7)

namespace scanner {

namespace {

class AVFifoBuffer;

typedef struct CuvidContext {
  CUvideodecoder cudecoder;
  CUvideoparser cuparser;

  AVBufferRef* hwdevice;
  AVBufferRef* hwframe;

  AVBSFContext* bsf;

  AVFifoBuffer* frame_queue;

  int internal_error;

  cudaVideoCodec codec_type;
  cudaVideoChromaFormat chroma_format;
} CuvidContext;
}

pthread_mutex_t av_mutex;

VideoSeparator::VideoSeparator(CUcontext cuda_context, AVCodecContext* cc)
    : cuda_context_(cuda_context),
      cc_(cc),
      parser_(nullptr),
      prev_frame_(0),
      is_metadata_(true),
      is_keyframe_(false),
      decode_time_(0) {
  CUcontext dummy;

  CUD_CHECK(cuCtxPushCurrent(cuda_context_));

  CUVIDPARSERPARAMS cuparseinfo = {};

  cuparseinfo.CodecType = cudaVideoCodec_H264;
  cuparseinfo.ulMaxNumDecodeSurfaces = 20;
  cuparseinfo.ulMaxDisplayDelay = 4;
  cuparseinfo.pUserData = this;
  cuparseinfo.pfnSequenceCallback = VideoSeparator::cuvid_handle_video_sequence;
  cuparseinfo.pfnDecodePicture = VideoSeparator::cuvid_handle_picture_decode;
  cuparseinfo.pfnDisplayPicture = VideoSeparator::cuvid_handle_picture_display;

  CUVIDEOFORMATEX cuparse_ext = {};
  cuparseinfo.pExtVideoInfo = &cuparse_ext;

  CuvidContext* ctx = reinterpret_cast<CuvidContext*>(cc->priv_data);
  if (cc->codec->id == AV_CODEC_ID_H264 || cc->codec->id == AV_CODEC_ID_HEVC) {
    cuparse_ext.format.seqhdr_data_length = ctx->bsf->par_out->extradata_size;
    memcpy(cuparse_ext.raw_seqhdr_data, ctx->bsf->par_out->extradata,
           FFMIN(sizeof(cuparse_ext.raw_seqhdr_data),
                 ctx->bsf->par_out->extradata_size));
  } else if (cc->extradata_size > 0) {
    cuparse_ext.format.seqhdr_data_length = cc->extradata_size;
    memcpy(cuparse_ext.raw_seqhdr_data, cc->extradata,
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

  AVPacket filter_packet = {0};
  AVPacket filtered_packet = {0};
  CUdeviceptr mapped_frame = 0;
  int ret = 0, eret = 0;

  CuvidContext* ctx = reinterpret_cast<CuvidContext*>(cc_->priv_data);
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
    memcpy(metadata_packets_.data() + prev_size, &packet->size, sizeof(int));
    memcpy(metadata_packets_.data() + prev_size + sizeof(int), packet->data,
           packet->size);
  } else {
    size_t prev_size = bitstream_packets_.size();
    bitstream_packets_.resize(prev_size + packet->size + sizeof(int));
    memcpy(bitstream_packets_.data() + prev_size, &packet->size, sizeof(int));
    memcpy(bitstream_packets_.data() + prev_size + sizeof(int), packet->data,
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

CUVIDDECODECREATEINFO VideoSeparator::get_decoder_info() {
  return decoder_info_;
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

int VideoSeparator::cuvid_handle_video_sequence(void* opaque,
                                                CUVIDEOFORMAT* format) {
  VideoSeparator& separator = *reinterpret_cast<VideoSeparator*>(opaque);

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

int VideoSeparator::cuvid_handle_picture_decode(void* opaque,
                                                CUVIDPICPARAMS* picparams) {
  VideoSeparator& separator = *reinterpret_cast<VideoSeparator*>(opaque);

  if (picparams->intra_pic_flag) {
    separator.is_keyframe_ = true;
  } else {
    separator.is_keyframe_ = false;
  }
  separator.prev_frame_ += 1;
}

int VideoSeparator::cuvid_handle_picture_display(
    void* opaque, CUVIDPARSERDISPINFO* dispinfo) {
  VideoSeparator& separator = *reinterpret_cast<VideoSeparator*>(opaque);
}
}
