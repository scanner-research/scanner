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
#include "scanner/api/op.h"
#include "scanner/util/memory.h"
#include "stdlib/stdlib.pb.h"

#include <cmath>

namespace scanner {

class BlurKernel : public Kernel, public VideoKernel {
 public:
  BlurKernel(const KernelConfig& config) : Kernel(config) {
    scanner::proto::BlurArgs args;
    bool parsed = args.ParseFromArray(config.args.data(), config.args.size());
    if (!parsed || config.args.size() == 0) {
      RESULT_ERROR(&valid_, "Could not parse BlurArgs");
      return;
    }

    kernel_size_ = args.kernel_size();
    sigma_ = args.sigma();

    filter_left_ = std::ceil(kernel_size_ / 2.0) - 1;
    filter_right_ = kernel_size_ / 2;

    valid_.set_success(true);
  }

  void validate(Result* result) override { result->CopyFrom(valid_); }

  void new_frame_info() {
    frame_width_ = frame_info_.width();
    frame_height_ = frame_info_.height();
  }

  void execute(const Elements& input_columns,
               Elements& output_columns) override {
    auto& frame_col = input_columns[0];
    check_frame(CPU_DEVICE, frame_col);

    i32 width = frame_width_;
    i32 height = frame_height_;
    size_t frame_size = width * height * 3 * sizeof(u8);
    FrameInfo info = frame_col.as_const_frame()->as_frame_info();
    Frame* output_frame = new_frame(CPU_DEVICE, info);

    const u8* frame_buffer = frame_col.as_const_frame()->data;
    u8* blurred_buffer = output_frame->data;
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
    insert_frame(output_columns[0], output_frame);
  }

 private:
  i32 kernel_size_;
  i32 filter_left_;
  i32 filter_right_;
  f64 sigma_;

  i32 frame_width_;
  i32 frame_height_;
  Result valid_;
};

REGISTER_OP(Blur).frame_input("frame").frame_output("frame");

REGISTER_KERNEL(Blur, BlurKernel).device(DeviceType::CPU).num_devices(1);
}
