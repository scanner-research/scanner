#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"
#include "tests/test_ops.pb.h"
#include <cmath>

namespace scanner {
namespace {
const i32 BINS = 16;
}

class HistogramKernelCPU : public BatchedKernel {
 public:
  HistogramKernelCPU(const KernelConfig& config)
    : BatchedKernel(config), device_(config.devices[0]) {}

  void execute(const BatchedElements& input_columns,
               BatchedElements& output_columns) override {
    auto& frame_col = input_columns[0];

    size_t hist_size = BINS * 3 * sizeof(int);
    i32 input_count = num_rows(frame_col);

    u8* output_block = new_block_buffer_size(device_, hist_size, input_count);

    for (i32 i = 0; i < input_count; ++i) {
      cv::Mat img = frame_to_mat(frame_col[i].as_const_frame());

      float range[] = {0, 256};
      const float* histRange = {range};

      u8* output_buf = output_block + i * hist_size;

      for (i32 j = 0; j < 3; ++j) {
        int channels[] = {j};
        cv::Mat hist;
        cv::calcHist(&img, 1, channels, cv::Mat(),
                     hist,
                     1, &BINS,
                     &histRange);
        cv::Mat out(BINS, 1, CV_32SC1, output_buf + j * BINS * sizeof(int));
        hist.convertTo(out, CV_32SC1);
      }

      insert_element(output_columns[0], output_buf, hist_size);
    }
  }

 private:
  DeviceHandle device_;
};

REGISTER_OP(Histogram).frame_input("frame").output("histogram", ColumnType::Bytes, "Histogram");

REGISTER_KERNEL(Histogram, HistogramKernelCPU)
    .device(DeviceType::CPU)
    .batch()
    .num_devices(1);



class OpticalFlowKernelCPU : public StenciledKernel, public VideoKernel {
 public:
  OpticalFlowKernelCPU(const KernelConfig& config)
    : StenciledKernel(config),
      device_(config.devices[0]) {
    flow_finder_ =
        cv::FarnebackOpticalFlow::create(3, 0.5, false, 15, 3, 5, 1.2, 0);
  }

  void new_frame_info() override {
    grayscale_.resize(0);
    for (i32 i = 0; i < 2; ++i) {
      grayscale_.emplace_back(frame_info_.height(), frame_info_.width(),
                              CV_8UC1);
    }
  }

  void execute(const StenciledElements& input_columns,
               Elements& output_columns) override {
    auto& frame_col = input_columns[0];
    check_frame(device_, frame_col[0]);

    FrameInfo out_frame_info(frame_info_.height(), frame_info_.width(), 2,
                             FrameType::F32);
    Frame* output_frame = new_frame(device_, out_frame_info);

    cv::Mat input0 = frame_to_mat(frame_col[0].as_const_frame());
    cv::Mat input1 = frame_to_mat(frame_col[1].as_const_frame());
    cv::cvtColor(input0, grayscale_[0], cv::COLOR_BGR2GRAY);
    cv::cvtColor(input1, grayscale_[1], cv::COLOR_BGR2GRAY);
    cv::Mat flow = frame_to_mat(output_frame);
    flow_finder_->calc(grayscale_[0], grayscale_[1], flow);
    insert_frame(output_columns[0], output_frame);
  }

 private:
  DeviceHandle device_;
  cv::Ptr<cv::DenseOpticalFlow> flow_finder_;
  std::vector<cv::Mat> grayscale_;
};

REGISTER_OP(OpticalFlow)
    .frame_input("frame")
    .frame_output("flow")
    .stencil({0, 1});

REGISTER_KERNEL(OpticalFlow, OpticalFlowKernelCPU)
    .device(DeviceType::CPU)
    .num_devices(1);


class ResizeKernel : public BatchedKernel {
 public:
  ResizeKernel(const KernelConfig& config)
    : BatchedKernel(config), device_(config.devices[0]) {
  }

  void new_stream(const std::vector<u8>& args) override {
    args_.ParseFromArray(args.data(), args.size());
  }

  void execute(const BatchedElements& input_columns,
               BatchedElements& output_columns) override {
    auto& frame_col = input_columns[0];

    const Frame* frame = frame_col[0].as_const_frame();

    i32 target_width = args_.width();
    i32 target_height = args_.height();
    if (args_.preserve_aspect()) {
      if (target_width == 0) {
        target_width =
            frame->width() * target_height / frame->height();
      } else {
        target_height =
            frame->height() * target_width / frame->width();
      }
    }
    if (args_.min()) {
      if (frame->width() <= target_width &&
          frame->height() <= target_height) {
        target_width = frame->width();
        target_height = frame->height();
      }
    }

    i32 input_count = num_rows(frame_col);
    FrameInfo info(target_height, target_width, frame->channels(), frame->type);
    std::vector<Frame*> output_frames = new_frames(device_, info, input_count);

    for (i32 i = 0; i < input_count; ++i) {
      cv::Mat img = frame_to_mat(frame_col[i].as_const_frame());
      cv::Mat out_mat = frame_to_mat(output_frames[i]);
      cv::resize(img, out_mat, cv::Size(target_width, target_height));
      insert_frame(output_columns[0], output_frames[i]);
    }
  }

 private:
  DeviceHandle device_;
  ResizeArgs args_;
  int interp_type_;
};

REGISTER_OP(Resize).frame_input("frame").frame_output("frame").stream_protobuf_name(
    "ResizeArgs");

REGISTER_KERNEL(Resize, ResizeKernel).device(DeviceType::CPU).batch().num_devices(1);


class TestIncrementKernel : public Kernel {
 public:
  TestIncrementKernel(const KernelConfig& config)
    : Kernel(config),
      device_(config.devices[0]) {}

  void reset() {
    next_int_ = 0;
  }

  void execute(const Elements& input_columns,
               Elements& output_columns) override {
    if (last_row_ + 1 != input_columns[0].index) {
      last_row_ = input_columns[0].index - 1;
      reset();
    }
    last_row_++;

    u8* buffer = new_buffer(device_, sizeof(i64));
    *((i64*)buffer) = next_int_++;
    insert_element(output_columns[0], buffer, sizeof(i64));
  }

 private:
  DeviceHandle device_;
  i64 next_int_ = 0;
  i64 last_row_ = 0;
};

REGISTER_OP(TestIncrementUnbounded)
.input("ignore")
.output("integer")
.unbounded_state();

REGISTER_OP(TestIncrementUnboundedFrame)
.frame_input("ignore")
.output("integer")
.unbounded_state();

REGISTER_KERNEL(TestIncrementUnbounded, TestIncrementKernel)
.device(DeviceType::CPU)
.num_devices(1);

REGISTER_KERNEL(TestIncrementUnboundedFrame, TestIncrementKernel)
.device(DeviceType::CPU)
.num_devices(1);

REGISTER_OP(TestIncrementBounded)
.input("ignore")
.output("integer")
.bounded_state();

REGISTER_OP(TestIncrementBoundedFrame)
.frame_input("ignore")
.output("integer")
.bounded_state();

REGISTER_KERNEL(TestIncrementBounded, TestIncrementKernel)
.device(DeviceType::CPU)
.num_devices(1);

REGISTER_KERNEL(TestIncrementBoundedFrame, TestIncrementKernel)
.device(DeviceType::CPU)
.num_devices(1);


class BlurKernel : public Kernel, public VideoKernel {
 public:
  BlurKernel(const KernelConfig& config) : Kernel(config) {
    BlurArgs args;
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

REGISTER_OP(Blur).frame_input("frame").frame_output("frame").protobuf_name(
    "BlurArgs");

REGISTER_KERNEL(Blur, BlurKernel).device(DeviceType::CPU).num_devices(1);


class SleepKernel : public Kernel {
 public:
  SleepKernel(const KernelConfig& config)
    : Kernel(config),
      device_(config.devices[0]) {}

  void execute(const Elements& input_columns, Elements& output_columns) override {
    sleep(2);
    insert_element(output_columns[0], new_buffer(device_, 1), 1);
  }

 private:
  DeviceHandle device_;
};

REGISTER_OP(Sleep).input("ignore").output("dummy");

REGISTER_OP(SleepFrame).frame_input("ignore").output("dummy");

REGISTER_KERNEL(Sleep, SleepKernel).device(DeviceType::CPU).num_devices(1);

REGISTER_KERNEL(Sleep, SleepKernel).device(DeviceType::GPU).num_devices(1);

REGISTER_KERNEL(SleepFrame, SleepKernel).device(DeviceType::CPU).num_devices(1);

REGISTER_KERNEL(SleepFrame, SleepKernel).device(DeviceType::GPU).num_devices(1);

}
