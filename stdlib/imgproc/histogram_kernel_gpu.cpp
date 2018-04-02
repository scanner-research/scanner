#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/cuda.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"

namespace scanner {
namespace {
const i32 BINS = 16;
}

class HistogramKernelGPU : public BatchedKernel, public VideoKernel {
 public:
  HistogramKernelGPU(const KernelConfig& config)
    : BatchedKernel(config),
      device_(config.devices[0]),
      num_cuda_streams_(32),
      streams_(num_cuda_streams_) {}

  void new_frame_info() override {
    set_device();
    streams_.resize(0);
    streams_.resize(num_cuda_streams_);
    planes_.clear();
    for (i32 i = 0; i < 3; ++i) {
      planes_.push_back(
          cvc::GpuMat(frame_info_.width(), frame_info_.height(), CV_8UC1));
    }
  }

  void execute(const BatchedElements& input_columns,
               BatchedElements& output_columns) override {
    auto& frame_col = input_columns[0];

    set_device();
    check_frame(device_, frame_col[0]);

    size_t hist_size = BINS * 3 * sizeof(float);
    i32 input_count = num_rows(frame_col);
    u8* output_block =
        new_block_buffer(device_, hist_size * input_count, input_count);

    for (i32 i = 0; i < input_count; ++i) {
      i32 sid = i % num_cuda_streams_;
      cv::cuda::Stream& s = streams_[sid];

      // TODO(wcrichto): implement correctly w/ streams
      cvc::GpuMat img = frame_to_gpu_mat(frame_col[i].as_const_frame());
      cvc::split(img, planes_);

      u8* output_buf = output_block + i * hist_size;
      cvc::GpuMat out_mat(1, BINS * 3, CV_32S, output_buf);

      for (i32 j = 0; j < 3; ++j) {
        cvc::histEven(planes_[j], out_mat(cv::Rect(j * BINS, 0, BINS, 1)), BINS,
                      0, 256);
      }

      insert_element(output_columns[0], output_buf, hist_size);
    }

    for (cv::cuda::Stream& s : streams_) {
      s.waitForCompletion();
    }
  }

  void set_device() {
    CUDA_PROTECT({ CU_CHECK(cudaSetDevice(device_.id)); });
    cvc::setDevice(device_.id);
  }

 private:
  DeviceHandle device_;
  i32 num_cuda_streams_;
  std::vector<cv::cuda::Stream> streams_;
  std::vector<cvc::GpuMat> planes_;
};

REGISTER_KERNEL(Histogram, HistogramKernelGPU)
    .device(DeviceType::GPU)
    .batch()
    .num_devices(1);
}
