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

}

pthread_mutex_t av_mutex;

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
  DatasetItemMetadata metadata)
  : max_output_frames_(32),
    max_mapped_frames_(8),
    streams_(max_mapped_frames_),
    cuda_context_(cuda_context),
    metadata_(metadata),
    metadata_packets_(metadata.metadata_packets),
    parser_(nullptr),
    decoder_(nullptr),
    mapped_frames_(max_mapped_frames_, 0),
    prev_frame_(0),
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
  while (pos < metadata_packets_.size()) {
    int encoded_packet_size =
      *reinterpret_cast<int*>(metadata_packets_.data() + pos);
    pos += sizeof(int);
    char* encoded_packet = metadata_packets_.data() + pos;
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
