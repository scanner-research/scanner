#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"

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
    cv::cvtColor(input0, grayscale_[0], CV_BGR2GRAY);
    cv::cvtColor(input1, grayscale_[1], CV_BGR2GRAY);
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

}
