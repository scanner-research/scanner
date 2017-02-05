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

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#include <glog/logging.h>

#ifdef HAVE_CUDA
#define CUDA_PROTECT(s) (s);
#else
#define CUDA_PROTECT(s) {                       \
    LOG(FATAL) << "Cuda not enabled.";          \
  }
#endif

#ifdef HAVE_CUDA

#define CU_CHECK(ans) \
  { cuAssert((ans), __FILE__, __LINE__); }

inline void cuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    LOG(FATAL) << "GPUassert: "
               << cudaGetErrorString(code) << " "
               << file << " "
               << line;
  }
}

#define CUD_CHECK(ans) \
  { cudAssert((ans), __FILE__, __LINE__); }

inline void cudAssert(CUresult code, const char *file, int line) {
  if (code != CUDA_SUCCESS) {
    const char *err_str;
    cuGetErrorString(code, &err_str);
    LOG(FATAL) << "GPUassert: "
               << err_str << " "
               << file << " "
               << line;
  }
}

#endif
