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

#include "scanner/kernels/blur_kernel.h"
#include "scanner/kernels/args.pb.h"

#include "scanner/api/evaluator.h"

#include <cmath>

namespace scanner {

BlurKernel::BlurKernel(const Kernel::Config& config)
    : Kernel(config) {
  BlurArgs args;
  args.ParseFromArray(config.args.data(), config.args.size());

  kernel_size_ = args.kernel_size();
  sigma_ = args.sigma();

  filter_left_ = std::ceil(kernel_size_ / 2.0) - 1;
  filter_right_ = kernel_size_ / 2;
}

void BlurKernel::execute(const BatchedColumns &input_columns,
                         BatchedColumns &output_columns) {
  i32 input_count = (i32)input_columns[0].rows.size();

  i32 width = frame_width_;
  i32 height = frame_height_;
  size_t frame_size = width * height * 3 * sizeof(u8);

  for (i32 i = 0; i < input_count; ++i) {
    u8* input_buffer = input_columns[0].rows[i].buffer;
    u8* output_buffer = new u8[frame_size];

    u8* frame_buffer = input_buffer;
    u8* blurred_buffer = (output_buffer);
    for (i32 y = filter_left_; y < height - filter_right_; ++y) {
      for (i32 x = filter_left_; x < width - filter_right_; ++x) {
        for (i32 c = 0; c < 3; ++c) {
          u32 value = 0;
          for (i32 ry = -filter_left_; ry < filter_right_ + 1; ++ry) {
            for (i32 rx = -filter_left_; rx < filter_right_ + 1; ++rx) {
              value += frame_buffer[(y + ry) * width * 3 + (x + rx) * 3 + c];
            }
          }
          blurred_buffer[y * width * 3 + x * 3 + c] =
              value / ((filter_right_ + filter_left_ + 1) *
                       (filter_right_ + filter_left_ + 1));
        }
      }
    }
    output_columns[0].rows.push_back(Row{output_buffer, frame_size});
  }
}

REGISTER_EVALUATOR(Blur).outputs({"frame", "frame_info"});

REGISTER_KERNEL(Blur, BlurKernel).device(DeviceType::CPU).num_devices(1);

}
