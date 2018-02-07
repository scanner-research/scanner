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
    u8* output_block =
        new_block_buffer(device_, hist_size * input_count, input_count);

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

REGISTER_OP(Histogram).frame_input("frame").output("histogram");

REGISTER_KERNEL(Histogram, HistogramKernelCPU)
    .device(DeviceType::CPU)
    .batch()
    .num_devices(1);
}

