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

#pragma once

#include "scanner/util/common.h"

#include <cstddef>

namespace scanner {

static const i64 DEFAULT_POOL_SIZE = 2L * 1024L * 1024L * 1024L;

void init_memory_allocators(MemoryPoolConfig config,
                            std::vector<i32> gpu_device_ids);

void destroy_memory_allocators();

u8* new_buffer(DeviceHandle device, size_t size);

u8* new_block_buffer(DeviceHandle device, size_t size, i32 refs);

void delete_buffer(DeviceHandle device, u8* buffer);

void memcpy_buffer(u8* dest_buffer, DeviceHandle dest_device,
                   const u8* src_buffer, DeviceHandle src_device, size_t size);

void memcpy_vec(std::vector<u8*> dest_buffers, DeviceHandle dest_device,
                const std::vector<u8*> src_buffers, DeviceHandle src_device,
                std::vector<size_t> sizes);
}
