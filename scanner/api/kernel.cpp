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

BaseKernel::BaseKernel(const KernelConfig& config) {}

StenciledBatchedKernel::StenciledBatchedKernel(const KernelConfig& config)
    : BaseKernel(config) {}

void StenciledBatchedKernel::execute_kernel(
    const StenciledBatchedElements& input_columns,
    BatchedElements& output_columns) {
  execute(input_columns, output_columns);
}

StenciledKernel::StenciledKernel(const KernelConfig& config)
  : BaseKernel(config) {}

void StenciledKernel::execute_kernel(
    const StenciledBatchedElements& input_columns,
    BatchedElements& output_columns) {
  StenciledElements in;
  for (auto& col : input_columns) {
    in.emplace_back();
    std::vector<Element>& b = in.back();
    b = col[0];
  }

  Columns out_cols(output_columns.size());
  execute(in, out_cols);
  for (size_t i = 0; i < out_cols.size(); ++i) {
    output_columns[i].push_back(out_cols[i]);
  }
}

BatchedKernel::BatchedKernel(const KernelConfig& config)
    : BaseKernel(config) {}

void BatchedKernel::execute_kernel(
    const StenciledBatchedElements& input_columns,
    BatchedElements& output_columns) {
  BatchedElements in;
  for (auto& col : input_columns) {
    in.emplace_back();
    std::vector<Element>& b = in.back();
    for (auto& stencil : col) {
      b.push_back(stencil[0]);
    }
  }

  execute(in, output_columns);
}

Kernel::Kernel(const KernelConfig& config)
    : BaseKernel(config) {}

void Kernel::execute_kernel(
    const StenciledBatchedElements& input_columns,
    BatchedElements& output_columns) {

  Columns in_cols;
  for (auto& col : input_columns) {
    in_cols.push_back(col[0][0]);
  }

  Columns out_cols(output_columns.size());
  execute(in_cols, out_cols);
  for (size_t i = 0; i < out_cols.size(); ++i) {
    output_columns[i].push_back(out_cols[i]);
  }
}

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
  memcpy_buffer((u8*)buffer, CPU_DEVICE, element.buffer, device, element.size);
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
  auto& input_devices = builder.input_devices_;
  auto& output_devices = builder.output_devices_;
  bool can_batch = builder.can_batch_;
  i32 preferred_batch = builder.preferred_batch_size_;
  KernelConstructor constructor = builder.constructor_;
  internal::KernelFactory* factory = new internal::KernelFactory(
      name, type, num_devices, input_devices, output_devices, can_batch,
      preferred_batch, constructor);
  internal::KernelRegistry* registry = internal::get_kernel_registry();
  registry->add_kernel(name, factory);
}
}
}
