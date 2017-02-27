#include "scanner/api/op.h"
#include "scanner/api/kernel.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"

namespace scanner {
namespace {
const i32 BINS = 16;
}

class HistogramKernelCPU : public VideoKernel {
public:
  HistogramKernelCPU(const Kernel::Config &config)
      : VideoKernel(config), device_(config.devices[0]) {
    assert(config.input_columns.size() == 2);
  }

  void execute(const BatchedColumns &input_columns,
               BatchedColumns &output_columns) override {
    check_frame_info(device_, input_columns[1]);

    size_t hist_size = BINS * 3 * sizeof(float);
    i32 input_count = input_columns[0].rows.size();
    u8 *output_block =
        new_block_buffer(device_, hist_size * input_count, input_count);

    cv::Mat tmp;
    for (i32 i = 0; i < input_count; ++i) {
      cv::Mat img(frame_info_.height(), frame_info_.width(), CV_8UC3,
                  (u8 *)input_columns[0].rows[i].buffer);

      float range[] = {0, 256};
      const float *histRange = {range};

      u8 *output_buf = output_block + i * hist_size;

      for (i32 j = 0; j < 3; ++j) {
        int channels[] = {j};
        cv::Mat out(BINS, 1, CV_32F , output_buf + BINS * sizeof(float));
        cv::calcHist(&img, 1, channels, cv::Mat(), out, 1, &BINS, &histRange);
        out.convertTo(out, CV_32S);
      }

      output_columns[0].rows.push_back(Row{output_buf, hist_size});
    }
  }

private:
  DeviceHandle device_;
};

REGISTER_OP(Histogram).inputs({"frame", "frame_info"}).outputs({"histogram"});

REGISTER_KERNEL(Histogram, HistogramKernelCPU)
    .device(DeviceType::CPU)
    .num_devices(1);
}
