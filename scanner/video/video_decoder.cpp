/* Copyright 2016 Carnegie Mellon University
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

#include "scanner/video/video_decoder.h"

#ifdef HAVE_NVIDIA_VIDEO_HARDWARE
#include "scanner/util/cuda.h"
#include "scanner/video/nvidia/nvidia_video_decoder.h"
#endif

#ifdef HAVE_INTEL_VIDEO_HARDWARE
#include "scanner/video/intel/intel_video_decoder.h"
#endif

#include "scanner/video/software/software_video_decoder.h"

#include <cassert>

namespace scanner {
namespace internal {

std::vector<VideoDecoderType> VideoDecoder::get_supported_decoder_types() {
  std::vector<VideoDecoderType> decoder_types;
#ifdef HAVE_NVIDIA_VIDEO_HARDWARE
  decoder_types.push_back(VideoDecoderType::NVIDIA);
#endif
#ifdef HAVE_INTEL_VIDEO_HARDWARE
  decoder_types.push_back(VideoDecoderType::INTEL);
#endif
  decoder_types.push_back(VideoDecoderType::SOFTWARE);

  return decoder_types;
}

bool VideoDecoder::has_decoder_type(VideoDecoderType type) {
  std::vector<VideoDecoderType> types =
      VideoDecoder::get_supported_decoder_types();

  for (const VideoDecoderType& supported_type : types) {
    if (type == supported_type) return true;
  }

  return false;
}

VideoDecoder* VideoDecoder::make_from_config(DeviceHandle device_handle,
                                             i32 num_devices,
                                             VideoDecoderType type) {
  VideoDecoder* decoder = nullptr;

  switch (type) {
    case VideoDecoderType::NVIDIA: {
#ifdef HAVE_NVIDIA_VIDEO_HARDWARE
      // HACK(apoms): we are just going to assume all processing is done in the
      //   default context for now and retain it ourselves. Ideally we would
      //   allow the user to pass in the CUcontext they want to use for
      //   decoding frames into but that would require providing opaque
      //   configuration data to this function which we are avoiding for now.
      //   The
      //   reason we are avoding it for now is that by having configuration data
      //   for different decoders, the client code ends up needing to do
      //   conditional includes depending on which decoders are available in
      //   order to fill in the configuration data, which is just messy.
      CUD_CHECK(cuInit(0));
      CUcontext cuda_context;
      CUD_CHECK(cuDevicePrimaryCtxRetain(&cuda_context, device_handle.id));

      decoder = new NVIDIAVideoDecoder(device_handle.id, device_handle.type,
                                       cuda_context);
#else
#endif
      break;
    }
    case VideoDecoderType::INTEL: {
#ifdef HAVE_INTEL_VIDEO_HARDWARE
      decoder = new IntelVideoDecoder(device_handle.id, device_handle.type);
#else
#endif
      break;
    }
    case VideoDecoderType::SOFTWARE: {
      decoder = new SoftwareVideoDecoder(device_handle.id, device_handle.type,
                                         num_devices);
      break;
    }
    default: {}
  }

  return decoder;
}

void VideoDecoder::set_profiler(Profiler* profiler) { profiler_ = profiler; }
}
}
