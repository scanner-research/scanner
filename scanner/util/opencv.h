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

#pragma once

#include <opencv2/opencv.hpp>

#ifdef HAVE_CUDA
#include <opencv2/core/cuda.hpp>

void convertNV12toRGBA(
  const cv::cuda::GpuMat& in,
  cv::cuda::GpuMat& outFrame,
  int width,
  int height,
  cv::cuda::Stream& stream = cv::cuda::Stream::Null());

void convertRGBInterleavedToPlanar(
  const cv::cuda::GpuMat& in,
  cv::cuda::GpuMat& outFrame,
  int width,
  int height,
  cv::cuda::Stream& stream = cv::cuda::Stream::Null());
#endif
