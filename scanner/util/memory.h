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

u8* new_buffer(DeviceType type, int device_id, size_t size);

void delete_buffer(DeviceType type, int device_id, u8* buffer);

void memcpy_buffer(u8* dest_buffer, DeviceType dest_type, i32 dest_device_id,
                   const u8* src_buffer, DeviceType src_type, i32 src_device_id,
                   size_t size);
}
