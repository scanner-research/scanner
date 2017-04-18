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

#include "scanner/api/kernel.h"
#include "scanner/engine/kernel_factory.h"
#include "scanner/engine/kernel_registry.h"
#include "scanner/util/memory.h"

namespace scanner {

Element::Element(u8* _buffer, size_t _size)
  : buffer(_buffer), size(_size), is_frame(false) {}

Element::Element(Frame* frame)
  : buffer((u8*)frame), size(sizeof(Frame)), is_frame(true) {}

Kernel::Kernel(const Config& config) {}

void VideoKernel::check_frame(const DeviceHandle& device,
                              const Element& element) {
  const Frame* frame = element.as_const_frame();
  bool same = (frame->type == frame_info_.type);
  for (i32 i = 0; i < 3; ++i) {
    same &= (frame->shape[i] == frame_info_.shape[i]);
  }
  if (!same) {
    memcpy(frame_info_.shape, frame->shape, sizeof(int) * 3);
    frame_info_.type = frame->type;
    new_frame_info();
  }
}

void VideoKernel::check_frame_info(const DeviceHandle& device,
                                   const Element& element) {
  // Assume that all the FrameInfos in the same batch are the same
  u8* buffer = new_buffer(CPU_DEVICE, element.size);
  memcpy_buffer((u8*)buffer, CPU_DEVICE, element.buffer, device,
                element.size);
  FrameInfo* frame_info = reinterpret_cast<FrameInfo*>(buffer);

  bool same = (frame_info->type == frame_info_.type);
  for (i32 i = 0; i < 3; ++i) {
    same &= (frame_info->shape[i] == frame_info_.shape[i]);
  }
  if (!same) {
    memcpy(frame_info_.shape, frame_info->shape, sizeof(int) * 3);
    frame_info_.type = frame_info->type;
    new_frame_info();
  }
  delete_buffer(CPU_DEVICE, buffer);
}

namespace internal {
KernelRegistration::KernelRegistration(const KernelBuilder& builder) {
  const std::string& name = builder.name_;
  DeviceType type = builder.device_type_;
  i32 num_devices = builder.num_devices_;
  KernelConstructor constructor = builder.constructor_;
  internal::KernelFactory* factory =
      new internal::KernelFactory(name, type, num_devices, 0, constructor);
  internal::KernelRegistry* registry = internal::get_kernel_registry();
  registry->add_kernel(name, factory);
}
}
}
