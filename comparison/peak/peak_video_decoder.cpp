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

#include "comparison/peak/peak_video_decoder.h"
#include "scanner/util/cuda.h"
#include "scanner/util/image.h"
#include "scanner/util/queue.h"

#include "storehouse/storage_backend.h"

#include <cassert>
#include <unistd.h>

#include <cuda.h>
#include <nvcuvid.h>

namespace scanner {

PeakVideoDecoder::PeakVideoDecoder(int device_id, DeviceType output_type,
                                       CUcontext cuda_context)
    : device_id_(device_id),
      output_type_(output_type),
      cuda_context_(cuda_context),
      streams_(max_mapped_frames_),
      parser_(nullptr),
      decoder_(nullptr),
      frame_queue_read_pos_(0),
      frame_queue_elements_(0),
      last_displayed_frame_(-1) {
  CUcontext dummy;

  CUD_CHECK(cuCtxPushCurrent(cuda_context_));

  for (int i = 0; i < max_mapped_frames_; ++i) {
    cudaStreamCreate(&streams_[i]);
    mapped_frames_[i] = 0;
  }
  for (i32 i = 0; i < max_output_frames_; ++i) {
    frame_in_use_[i] = false;
    undisplayed_frames_[i] = false;
    invalid_frames_[i] = false;
  }
}

PeakVideoDecoder::~PeakVideoDecoder() {
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

  // HACK(apoms): We are only using the primary context right now instead of
  //   allowing the user to specify their own CUcontext. Thus we need to release
  //   the primary context we retained when using the factory function to create
  //   this object (see VideoDecoder::make_from_config).
  CUD_CHECK(cuDevicePrimaryCtxRelease(device_id_));
}

void PeakVideoDecoder::configure(const InputFormat& metadata) {
  metadata_ = metadata;

  CUcontext dummy;

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

  CUVIDPARSERPARAMS cuparseinfo = {};
  // cuparseinfo.CodecType = metadata.codec_type;
  cuparseinfo.CodecType = cudaVideoCodec_H264;
  cuparseinfo.ulMaxNumDecodeSurfaces = max_output_frames_;
  cuparseinfo.ulMaxDisplayDelay = 1;
  cuparseinfo.pUserData = this;
  cuparseinfo.pfnSequenceCallback =
      PeakVideoDecoder::cuvid_handle_video_sequence;
  cuparseinfo.pfnDecodePicture =
      PeakVideoDecoder::cuvid_handle_picture_decode;
  cuparseinfo.pfnDisplayPicture =
      PeakVideoDecoder::cuvid_handle_picture_display;

  CUD_CHECK(cuvidCreateVideoParser(&parser_, &cuparseinfo));

  CUVIDDECODECREATEINFO cuinfo = {};
  // cuinfo.CodecType = metadata.codec_type;
  cuinfo.CodecType = cudaVideoCodec_H264;
  // cuinfo.ChromaFormat = metadata.chroma_format;
  cuinfo.ChromaFormat = cudaVideoChromaFormat_420;
  cuinfo.OutputFormat = cudaVideoSurfaceFormat_NV12;

  cuinfo.ulWidth = metadata.width();
  cuinfo.ulHeight = metadata.height();
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
    u8* encoded_packet = (u8*)(metadata_packets_.data() + pos);
    pos += encoded_packet_size;

    feed(encoded_packet, encoded_packet_size);
  }
}

bool PeakVideoDecoder::feed(const u8* encoded_buffer, size_t encoded_size,
                              bool discontinuity) {
  CUD_CHECK(cuCtxPushCurrent(cuda_context_));

  if (discontinuity) {
    CUVIDSOURCEDATAPACKET cupkt = {};
    cupkt.flags |= CUVID_PKT_DISCONTINUITY;
    CUD_CHECK(cuvidParseVideoData(parser_, &cupkt));

    last_displayed_frame_ = -1;
    for (i32 i = 0; i < max_output_frames_; ++i) {
      invalid_frames_[i] = undisplayed_frames_[i];
      undisplayed_frames_[i] = false;
    }
    // Empty queue because we have a new section of frames
    std::unique_lock<std::mutex> lock(frame_queue_mutex_);
    while (frame_queue_elements_ > 0) {
      const auto &dispinfo = frame_queue_[frame_queue_read_pos_];
      frame_in_use_[dispinfo.picture_index] = false;
      frame_queue_read_pos_ = (frame_queue_read_pos_ + 1) % max_output_frames_;
      frame_queue_elements_--;
    }
  }
  CUVIDSOURCEDATAPACKET cupkt = {};
  cupkt.payload_size = encoded_size;
  cupkt.payload = reinterpret_cast<const uint8_t*>(encoded_buffer);
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
      u8* encoded_packet = (u8*)(metadata_packets_.data() + pos);
      pos += encoded_packet_size;

      feed(encoded_packet, encoded_packet_size);
    }
  }

  CUcontext dummy;
  CUD_CHECK(cuCtxPopCurrent(&dummy));

  return frame_queue_elements_ > 0;
}

bool PeakVideoDecoder::discard_frame() {
  std::unique_lock<std::mutex> lock(frame_queue_mutex_);
  CUD_CHECK(cuCtxPushCurrent(cuda_context_));

  if (frame_queue_elements_ > 0) {
    const auto& dispinfo = frame_queue_[frame_queue_read_pos_];
    frame_in_use_[dispinfo.picture_index] = false;
    frame_queue_read_pos_ = (frame_queue_read_pos_ + 1) % max_output_frames_;
    frame_queue_elements_--;
  }

  CUcontext dummy;
  CUD_CHECK(cuCtxPopCurrent(&dummy));

  return frame_queue_elements_ > 0;
}

bool PeakVideoDecoder::get_frame(u8* decoded_buffer, size_t decoded_size) {
  std::unique_lock<std::mutex> lock(frame_queue_mutex_);
  CUD_CHECK(cuCtxPushCurrent(cuda_context_));
  if (frame_queue_elements_ > 0) {
    CUVIDPARSERDISPINFO dispinfo = frame_queue_[frame_queue_read_pos_];
    frame_queue_read_pos_ = (frame_queue_read_pos_ + 1) % max_output_frames_;
    frame_queue_elements_--;
    lock.unlock();

    CUVIDPROCPARAMS params = {};
    params.progressive_frame = dispinfo.progressive_frame;
    params.second_field = 0;
    params.top_field_first = dispinfo.top_field_first;

    int mapped_frame_index = dispinfo.picture_index % max_mapped_frames_;
    auto start_map = now();
    unsigned int pitch = 0;
    CUD_CHECK(cuvidMapVideoFrame(decoder_, dispinfo.picture_index,
                                 &mapped_frames_[mapped_frame_index], &pitch,
                                 &params));
    // cuvidMapVideoFrame does not wait for convert kernel to finish so sync
    // TODO(apoms): make this an event insertion and have the async 2d memcpy
    //              depend on the event
    if (profiler_) {
      profiler_->add_interval("map_frame", start_map, now());
    }
    CUdeviceptr mapped_frame = mapped_frames_[mapped_frame_index];
    CU_CHECK(convertNV12toRGBA((const u8 *)mapped_frame, pitch, decoded_buffer,
                               metadata_.width() * 3, metadata_.width(),
                               metadata_.height(), 0));

    CUD_CHECK(
        cuvidUnmapVideoFrame(decoder_, mapped_frames_[mapped_frame_index]));
    mapped_frames_[mapped_frame_index] = 0;

    frame_in_use_[dispinfo.picture_index] = false;
  }

  CUcontext dummy;
  CUD_CHECK(cuCtxPopCurrent(&dummy));

  return frame_queue_elements_;
}

int PeakVideoDecoder::decoded_frames_buffered() {
  return frame_queue_elements_;
}

void PeakVideoDecoder::wait_until_frames_copied() {
}

int PeakVideoDecoder::cuvid_handle_video_sequence(void* opaque,
                                                    CUVIDEOFORMAT* format) {
  PeakVideoDecoder& decoder = *reinterpret_cast<PeakVideoDecoder*>(opaque);
  return 1;
}

int PeakVideoDecoder::cuvid_handle_picture_decode(void* opaque,
                                                    CUVIDPICPARAMS* picparams) {
  PeakVideoDecoder& decoder = *reinterpret_cast<PeakVideoDecoder*>(opaque);

  int mapped_frame_index = picparams->CurrPicIdx;
  while (decoder.frame_in_use_[picparams->CurrPicIdx]) {
    usleep(500);
  };
  decoder.undisplayed_frames_[picparams->CurrPicIdx] = true;

  CUresult result = cuvidDecodePicture(decoder.decoder_, picparams);
  CUD_CHECK(result);

  return result == CUDA_SUCCESS;
}

int PeakVideoDecoder::cuvid_handle_picture_display(
    void* opaque, CUVIDPARSERDISPINFO* dispinfo) {
  PeakVideoDecoder& decoder = *reinterpret_cast<PeakVideoDecoder*>(opaque);
  if (!decoder.invalid_frames_[dispinfo->picture_index]) {
    decoder.frame_in_use_[dispinfo->picture_index] = true;
    while (true) {
      std::unique_lock<std::mutex> lock(decoder.frame_queue_mutex_);
      if (decoder.frame_queue_elements_ < max_output_frames_) {
        int write_pos =
            (decoder.frame_queue_read_pos_ + decoder.frame_queue_elements_) %
            max_output_frames_;
        decoder.frame_queue_[write_pos] = *dispinfo;
        decoder.frame_queue_elements_++;
        decoder.last_displayed_frame_++;
        break;
      }
      usleep(1000);
    }
  } else {
    decoder.invalid_frames_[dispinfo->picture_index] = false;
  }
  decoder.undisplayed_frames_[dispinfo->picture_index] = false;
  return true;
}
}
