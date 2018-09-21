/* Copyright 2018 Carnegie Mellon University
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

#include "scanner/util/common.h"

#include <cstddef>

namespace scanner {

static const i64 DEFAULT_POOL_SIZE = 2L * 1024L * 1024L * 1024L;

typedef struct {
  u8* buffer;
  size_t size;
  i32 refs;
  std::string call_file;
  i32 call_line;
} Allocation;

void init_memory_allocators(MemoryPoolConfig config,
                            std::vector<i32> gpu_device_ids);

void destroy_memory_allocators();

u8* new_buffer_(DeviceHandle device, size_t size, std::string call_file,
               i32 call_line);

#define new_buffer(device__, size__) \
  new_buffer_(device__, size__, __FILE__, __LINE__)

u8* new_block_buffer_(DeviceHandle device, size_t size, i32 refs,
                     std::string call_file, i32 call_line);

#define new_block_buffer(device__, size__, refs__)                \
  new_block_buffer_(device__, size__, refs__, __FILE__, __LINE__)

u8* new_block_buffer_sizes_(DeviceHandle device, const std::vector<size_t>& sizes,
                            std::string call_file, i32 call_line);

#define new_block_buffer_sizes(device__, sizes__)                 \
  new_block_buffer_sizes_(device__, sizes__, __FILE__, __LINE__)

void add_buffer_ref(DeviceHandle device, u8* buffer);

void add_buffer_refs(DeviceHandle device, u8* buffer, i32 refs);

void delete_buffer(DeviceHandle device, u8* buffer);

void memcpy_buffer(u8* dest_buffer, DeviceHandle dest_device,
                   const u8* src_buffer, DeviceHandle src_device, size_t size);

void memcpy_vec(std::vector<u8*>& dest_buffers, DeviceHandle dest_device,
                const std::vector<u8*>& src_buffers, DeviceHandle src_device,
                const std::vector<size_t>& sizes);

void copy_or_ref_buffers(std::vector<u8*>& dest_buffers,
                         DeviceHandle dest_device,
                         const std::vector<u8*>& src_buffers,
                         DeviceHandle src_device,
                         const std::vector<size_t>& sizes);

u64 current_memory_allocated(DeviceHandle device);

u64 max_memory_allocated(DeviceHandle device);

const std::vector<Allocation> allocator_allocations(DeviceHandle device);

}
