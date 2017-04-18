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

#include "scanner/video/nvidia/nvidia_video_decoder.h"
#include "scanner/util/cuda.h"
#include "scanner/util/image.h"
#include "scanner/util/queue.h"

#include "storehouse/storage_backend.h"

#include <cassert>
#include <thread>

#include <cuda.h>
#include <nvcuvid.h>

namespace scanner {
namespace internal {

NVIDIAVideoDecoder::NVIDIAVideoDecoder(int device_id, DeviceType output_type,
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

NVIDIAVideoDecoder::~NVIDIAVideoDecoder() {
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

void NVIDIAVideoDecoder::configure(const FrameInfo& metadata) {
  frame_width_ = metadata.shape[1];
  frame_height_ = metadata.shape[2];

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

  for (int i = 0; i < max_mapped_frames_; ++i) {
    mapped_frames_[i] = 0;
  }
  for (i32 i = 0; i < max_output_frames_; ++i) {
    frame_in_use_[i] = false;
    undisplayed_frames_[i] = false;
    invalid_frames_[i] = false;
  }
  frame_queue_read_pos_ = 0;
  frame_queue_elements_ = 0;

  CUVIDPARSERPARAMS cuparseinfo = {};
  // cuparseinfo.CodecType = metadata.codec_type;
  cuparseinfo.CodecType = cudaVideoCodec_H264;
  cuparseinfo.ulMaxNumDecodeSurfaces = max_output_frames_;
  cuparseinfo.ulMaxDisplayDelay = 1;
  cuparseinfo.pUserData = this;
  cuparseinfo.pfnSequenceCallback =
    NVIDIAVideoDecoder::cuvid_handle_video_sequence;
  cuparseinfo.pfnDecodePicture =
    NVIDIAVideoDecoder::cuvid_handle_picture_decode;
  cuparseinfo.pfnDisplayPicture =
    NVIDIAVideoDecoder::cuvid_handle_picture_display;

  CUD_CHECK(cuvidCreateVideoParser(&parser_, &cuparseinfo));

  CUVIDDECODECREATEINFO cuinfo = {};
  // cuinfo.CodecType = metadata.codec_type;
  cuinfo.CodecType = cudaVideoCodec_H264;
  // cuinfo.ChromaFormat = metadata.chroma_format;
  cuinfo.ChromaFormat = cudaVideoChromaFormat_420;
  cuinfo.OutputFormat = cudaVideoSurfaceFormat_NV12;

  cuinfo.ulWidth = frame_width_;
  cuinfo.ulHeight = frame_height_;
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

bool NVIDIAVideoDecoder::feed(const u8* encoded_buffer, size_t encoded_size,
                              bool discontinuity) {
  CUD_CHECK(cuCtxPushCurrent(cuda_context_));

  if (discontinuity) {
    {
      std::unique_lock<std::mutex> lock(frame_queue_mutex_);
      while (frame_queue_elements_ > 0) {
        const auto& dispinfo = frame_queue_[frame_queue_read_pos_];
        frame_in_use_[dispinfo.picture_index] = false;
        frame_queue_read_pos_ =
          (frame_queue_read_pos_ + 1) % max_output_frames_;
        frame_queue_elements_--;
      }
    }

    CUVIDSOURCEDATAPACKET cupkt = {};
    cupkt.flags |= CUVID_PKT_DISCONTINUITY;
    CUD_CHECK(cuvidParseVideoData(parser_, &cupkt));

    std::unique_lock<std::mutex> lock(frame_queue_mutex_);
    last_displayed_frame_ = -1;
    // Empty queue because we have a new section of frames
    for (i32 i = 0; i < max_output_frames_; ++i) {
      invalid_frames_[i] = undisplayed_frames_[i];
      undisplayed_frames_[i] = false;
    }
    while (frame_queue_elements_ > 0) {
      const auto& dispinfo = frame_queue_[frame_queue_read_pos_];
      frame_in_use_[dispinfo.picture_index] = false;
      frame_queue_read_pos_ = (frame_queue_read_pos_ + 1) % max_output_frames_;
      frame_queue_elements_--;
    }

    return false;
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

bool NVIDIAVideoDecoder::discard_frame() {
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

bool NVIDIAVideoDecoder::get_frame(u8* decoded_buffer, size_t decoded_size) {
  auto start = now();
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
    CU_CHECK(convertNV12toRGBA((const u8*)mapped_frame, pitch, decoded_buffer,
                               frame_width_ * 3, frame_width_, frame_height_,
                               0));
    CU_CHECK(cudaDeviceSynchronize());

    CUD_CHECK(
      cuvidUnmapVideoFrame(decoder_, mapped_frames_[mapped_frame_index]));
    mapped_frames_[mapped_frame_index] = 0;

    std::unique_lock<std::mutex> lock(frame_queue_mutex_);
    frame_in_use_[dispinfo.picture_index] = false;
  }

  CUcontext dummy;
  CUD_CHECK(cuCtxPopCurrent(&dummy));

  if (profiler_) {
    profiler_->add_interval("get_frame", start, now());
  }

  return frame_queue_elements_;
}

int NVIDIAVideoDecoder::decoded_frames_buffered() {
  return frame_queue_elements_;
}

void NVIDIAVideoDecoder::wait_until_frames_copied() {}

int NVIDIAVideoDecoder::cuvid_handle_video_sequence(void* opaque,
                                                    CUVIDEOFORMAT* format) {
  NVIDIAVideoDecoder& decoder = *reinterpret_cast<NVIDIAVideoDecoder*>(opaque);
  return 1;
}

int NVIDIAVideoDecoder::cuvid_handle_picture_decode(void* opaque,
                                                    CUVIDPICPARAMS* picparams) {
  NVIDIAVideoDecoder& decoder = *reinterpret_cast<NVIDIAVideoDecoder*>(opaque);

  int mapped_frame_index = picparams->CurrPicIdx;
  while (decoder.frame_in_use_[picparams->CurrPicIdx]) {
    usleep(500);
  };
  std::unique_lock<std::mutex> lock(decoder.frame_queue_mutex_);
  decoder.undisplayed_frames_[picparams->CurrPicIdx] = true;

  CUresult result = cuvidDecodePicture(decoder.decoder_, picparams);
  CUD_CHECK(result);

  return result == CUDA_SUCCESS;
}

int NVIDIAVideoDecoder::cuvid_handle_picture_display(
  void* opaque, CUVIDPARSERDISPINFO* dispinfo) {
  NVIDIAVideoDecoder& decoder = *reinterpret_cast<NVIDIAVideoDecoder*>(opaque);
  if (!decoder.invalid_frames_[dispinfo->picture_index]) {
    {
      std::unique_lock<std::mutex> lock(decoder.frame_queue_mutex_);
      decoder.frame_in_use_[dispinfo->picture_index] = true;
    }
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
    std::unique_lock<std::mutex> lock(decoder.frame_queue_mutex_);
    decoder.invalid_frames_[dispinfo->picture_index] = false;
  }
  std::unique_lock<std::mutex> lock(decoder.frame_queue_mutex_);
  decoder.undisplayed_frames_[dispinfo->picture_index] = false;
  return true;
}
}
}
