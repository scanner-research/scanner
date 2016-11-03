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

#include "scanner/util/memory.h"

#include <cassert>

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "scanner/util/cuda.h"
#endif

namespace scanner {

u8* new_buffer(DeviceType type, int device_id, size_t size) {
  assert(size > 0);
  u8* buffer = nullptr;
  if (type == DeviceType::CPU) {
    buffer = new u8[size];
  }
#ifdef HAVE_CUDA
  else if (type == DeviceType::GPU) {
    CU_CHECK(cudaSetDevice(device_id));
    CU_CHECK(cudaMalloc((void**)&buffer, size));
  }
#endif
  else {
    LOG(FATAL) << "Tried to allocate buffer of unsupported device type";
  }
  return buffer;
}

void delete_buffer(DeviceType type, int device_id, u8* buffer) {
  assert(buffer != nullptr);
  if (type == DeviceType::CPU) {
    delete[] buffer;
  }
#ifdef HAVE_CUDA
  else if (type == DeviceType::GPU) {
    CU_CHECK(cudaSetDevice(device_id));
    CU_CHECK(cudaFree(buffer));
  }
#endif
  else {
    LOG(FATAL) << "Tried to delete buffer of unsupported device type";
  }
  buffer = nullptr;
}

void memcpy_buffer(u8* dest_buffer, DeviceType dest_type, i32 dest_device_id,
                   const u8* src_buffer, DeviceType src_type, i32 src_device_id,
                   size_t size) {
#ifdef HAVE_CUDA
  CU_CHECK(cudaMemcpy(dest_buffer, src_buffer, size, cudaMemcpyDefault));
#else
  assert(dest_type == DeviceType::CPU);
  assert(dest_type == src_type);
  memcpy(dest_buffer, src_buffer, size);
#endif
}
}
