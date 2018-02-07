/* Copyright 2017 Carnegie Mellon University
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
#include "scanner/api/op.h"
#include "scanner/util/memory.h"
#include "stdlib/stdlib.pb.h"

#include <cmath>

namespace scanner {

class FrameDifferenceKernel : public StenciledKernel {
 public:
  BlurKernel(const KernelConfig& config) : StenciledKernel(config) {
    valid_.set_success(true);
  }

  void validate(Result* result) override {
    result->CopyFrom(valid_);
  }

  void new_frame_info() {
    frame_width_ = frame_info_.width();
    frame_height_ = frame_info_.height();
  }

  void execute(const StenciledElements& input_columns,
               Elements& output_columns) override {
    auto& frame_col = input_columns[0];
    check_frame(CPU_DEVICE, frame_col);

    FrameInfo info = frame_col.as_const_frame()->as_frame_info();
    i32 width = info.width();
    i32 height = info.height();
    i32 channels = info.channels();
    size_t frame_size = width * height * channels * sizeof(u8);

    const u8* secondary_frame_buffer = frame_col[0].as_const_frame()->data;
    const u8* primary_frame_buffer = frame_col[1].as_const_frame()->data;

    Frame* output_frame = new_frame(CPU_DEVICE, info);
    u8* output_buffer = output_frame->data;
    for (i32 y = 0; y < height; ++y) {
      for (i32 x = 0; x < width; ++x) {
        for (i32 c = 0; c < channels; ++c) {
          i64 offset = y * width * channels + width * channels + c;
          output_buffer[offset] =
              primary_frame_buffer[offset] - secondary_frame_buffer[offset]
        }
      }
    }
    insert_frame(output_columns[0], output_frame);
  }

 private:
  i32 frame_width_;
  i32 frame_height_;
  Result valid_;
};

REGISTER_OP(FrameDifference).frame_input("frame").frame_output("frame");

REGISTER_KERNEL(FrameDifference, FrameDifferenceKernel)
.device(DeviceType::CPU)
.batch();
.stencil({-1, 0});
.num_devices(1);
}
