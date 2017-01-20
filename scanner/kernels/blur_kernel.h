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

#include "scanner/api/kernel.h"

namespace scanner {

class BlurKernel : public Kernel {
 public:
  BlurKernel(Kernel::Config& config);

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override;

 private:
  i32 kernel_size_;
  i32 filter_left_;
  i32 filter_right_;
  f64 sigma_;

  i32 frame_width_;
  i32 frame_height_;
};

REGISTER_EVALUATOR(Blur).outputs({"frame", "frame_info"});

REGISTER_KERNEL(Blur, BlurKernelCPU).device(DeviceType::CPU).num_devices(1);

}
