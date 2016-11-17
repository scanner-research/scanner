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

#include "scanner/util/image.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace scanner {
// Taken from https://github.com/opencv/opencv/blob/master/modules/cudacodec/src/cuda/nv12_to_rgb.cu
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

namespace
{
    __constant__ float constHueColorSpaceMat[9] = {1.1644f, 0.0f, 1.596f, 1.1644f, -0.3918f, -0.813f, 1.1644f, 2.0172f, 0.0f};

    __device__ static void YUV2RGB(const uint* yuvi, float* red, float* green, float* blue)
    {
        float luma, chromaCb, chromaCr;

        // Prepare for hue adjustment
        luma     = (float)yuvi[0];
        chromaCb = (float)((int)yuvi[1] - 512.0f);
        chromaCr = (float)((int)yuvi[2] - 512.0f);

       // Convert YUV To RGB with hue adjustment
       *red   = (luma     * constHueColorSpaceMat[0]) +
                (chromaCb * constHueColorSpaceMat[1]) +
                (chromaCr * constHueColorSpaceMat[2]);

       *green = (luma     * constHueColorSpaceMat[3]) +
                (chromaCb * constHueColorSpaceMat[4]) +
                (chromaCr * constHueColorSpaceMat[5]);

       *blue  = (luma     * constHueColorSpaceMat[6]) +
                (chromaCb * constHueColorSpaceMat[7]) +
                (chromaCr * constHueColorSpaceMat[8]);
    }

    __device__ static void RGB_pack_10bit(float red, float green, float blue,
                                          u8* rgb) {
      // Clamp final 10 bit results
      red = ::fmin(::fmax(red, 0.0f), 1023.f);
      green = ::fmin(::fmax(green, 0.0f), 1023.f);
      blue = ::fmin(::fmax(blue, 0.0f), 1023.f);

      rgb[0] = ((uint)red) >> 2;
      rgb[1] = ((uint)green) >> 2;
      rgb[2] = ((uint)blue) >> 2;
    }

    // CUDA kernel for outputing the final ARGB output from NV12

    #define COLOR_COMPONENT_BIT_SIZE 10
    #define COLOR_COMPONENT_MASK     0x3FF

    __global__ void NV12_to_RGB(const u8 *srcImage, size_t nSourcePitch,
                                u8 *dstImage, size_t nDestPitch, uint width,
                                uint height) {
      // Pad borders with duplicate pixels, and we multiply by 2 because we
      // process 2 pixels per thread
      const int x = blockIdx.x * (blockDim.x << 1) + (threadIdx.x << 1);
      const int y = blockIdx.y * blockDim.y + threadIdx.y;

      if (x >= width || y >= height)
        return;

      // Read 2 Luma components at a time, so we don't waste processing since
      // CbCr are decimated this way.
      // if we move to texture we could read 4 luminance values

      uint yuv101010Pel[2];

      yuv101010Pel[0] = (srcImage[y * nSourcePitch + x]) << 2;
      yuv101010Pel[1] = (srcImage[y * nSourcePitch + x + 1]) << 2;

      const size_t chromaOffset = nSourcePitch * height;

      const int y_chroma = y >> 1;

      if (y & 1) // odd scanline ?
      {
        uint chromaCb = srcImage[chromaOffset + y_chroma * nSourcePitch + x];
        uint chromaCr =
            srcImage[chromaOffset + y_chroma * nSourcePitch + x + 1];

        if (y_chroma < ((height >> 1) - 1)) // interpolate chroma vertically
        {
          chromaCb =
              (chromaCb +
               srcImage[chromaOffset + (y_chroma + 1) * nSourcePitch + x] +
               1) >>
              1;
          chromaCr =
              (chromaCr +
               srcImage[chromaOffset + (y_chroma + 1) * nSourcePitch + x + 1] +
               1) >>
              1;
        }

        yuv101010Pel[0] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE + 2));
        yuv101010Pel[0] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));

        yuv101010Pel[1] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE + 2));
        yuv101010Pel[1] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
      } else {
        yuv101010Pel[0] |=
            ((uint)srcImage[chromaOffset + y_chroma * nSourcePitch + x]
             << (COLOR_COMPONENT_BIT_SIZE + 2));
        yuv101010Pel[0] |=
            ((uint)srcImage[chromaOffset + y_chroma * nSourcePitch + x + 1]
             << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));

        yuv101010Pel[1] |=
            ((uint)srcImage[chromaOffset + y_chroma * nSourcePitch + x]
             << (COLOR_COMPONENT_BIT_SIZE + 2));
        yuv101010Pel[1] |=
            ((uint)srcImage[chromaOffset + y_chroma * nSourcePitch + x + 1]
             << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
      }

      // this steps performs the color conversion
      uint yuvi[6];
      float red[2], green[2], blue[2];

      yuvi[0] = (yuv101010Pel[0] & COLOR_COMPONENT_MASK);
      yuvi[1] = ((yuv101010Pel[0] >> COLOR_COMPONENT_BIT_SIZE) &
                 COLOR_COMPONENT_MASK);
      yuvi[2] = ((yuv101010Pel[0] >> (COLOR_COMPONENT_BIT_SIZE << 1)) &
                 COLOR_COMPONENT_MASK);

      yuvi[3] = (yuv101010Pel[1] & COLOR_COMPONENT_MASK);
      yuvi[4] = ((yuv101010Pel[1] >> COLOR_COMPONENT_BIT_SIZE) &
                 COLOR_COMPONENT_MASK);
      yuvi[5] = ((yuv101010Pel[1] >> (COLOR_COMPONENT_BIT_SIZE << 1)) &
                 COLOR_COMPONENT_MASK);

      // YUV to RGB Transformation conversion
      YUV2RGB(&yuvi[0], &red[0], &green[0], &blue[0]);
      YUV2RGB(&yuvi[3], &red[1], &green[1], &blue[1]);

      // Clamp the results to RGB

      RGB_pack_10bit(red[0], green[0], blue[0],
                     &dstImage[y * nDestPitch + x * 3]);
      RGB_pack_10bit(red[1], green[1], blue[1],
                     &dstImage[y * nDestPitch + (x + 1) * 3]);
    }

__global__ void RGB_interleaved_to_planar(
  const u8* srcImage, size_t nSourcePitch,
  u8* dstImage, size_t nDestPitch,
  uint width, uint height)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const int channel_size  = (nDestPitch * height);
  dstImage[channel_size * 0 + y * nDestPitch + x] =
    srcImage[y * nSourcePitch + x * 3 + 0];
  dstImage[channel_size * 1 + y * nDestPitch + x] =
    srcImage[y * nSourcePitch + x * 3 + 1];
  dstImage[channel_size * 2 + y * nDestPitch + x] =
    srcImage[y * nSourcePitch + x * 3 + 2];
}

}

__host__ __device__ __forceinline__ int divUp(int total, int grain) {
    return (total + grain - 1) / grain;
}
// End OpenCV code

cudaError_t convertNV12toRGBA(const u8 *in, size_t in_pitch,
                              u8 *out, size_t out_pitch,
                              int width, int height,
                              cudaStream_t stream) {
  dim3 block(32, 8);
  dim3 grid(divUp(width, 2 * block.x), divUp(height, block.y));

  NV12_to_RGB<<<grid, block, 0, stream>>>(
      in, in_pitch, out, out_pitch, width, height);
  return cudaPeekAtLastError();
}

cudaError_t convertRGBInterleavedToPlanar(const u8 *in, size_t in_pitch,
                                          u8 *out, size_t out_pitch, int width,
                                          int height, cudaStream_t stream) {
  dim3 block(32, 8);
  dim3 grid(divUp(width, block.x), divUp(height, block.y));

  RGB_interleaved_to_planar<<<grid, block, 0, stream>>>(
      in, in_pitch, out, out_pitch, width, height);
  return cudaPeekAtLastError();
}

}
