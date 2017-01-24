#include "scanner/api/evaluator.h"
#include "scanner/api/kernel.h"
#include "scanner/util/opencv.h"
#include "scanner/util/memory.h"
#include "scanner/util/cuda.h"

namespace scanner {

const i32 BINS = 16;

class HistogramKernelCPU : public VideoKernel {
public:
  HistogramKernelCPU(const Kernel::Config& config)
    : VideoKernel(config), device_(config.devices[0]) {
    assert(config.input_columns.size() == 2);
  }

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    check_frame_info(device_, input_columns[1]);

    size_t hist_size = BINS * 3 * sizeof(float);
    i32 input_count = input_columns[0].rows.size();
    u8* output_block = new_block_buffer(
      device_, hist_size * input_count, input_count);

    cv::Mat tmp;
    for (i32 i = 0; i < input_count; ++i) {
      cv::Mat img(frame_info_.height, frame_info_.width, CV_8UC3,
                  (u8*) input_columns[0].rows[i].buffer);

      float range[] = {0, 256};
      const float* histRange = {range};

      u8* output_buf = output_block + i * hist_size;

      for (i32 j = 0; j < 3; ++j) {
        int channels[] = {j};
        cv::Mat out(BINS, 1, CV_32S, output_buf + BINS * sizeof(float));
        cv::calcHist(&img, 1, channels, cv::Mat(),
                     out,
                     1, &BINS,
                     &histRange);
      }

      output_columns[0].rows.push_back(Row{output_buf, hist_size});
    }
  }

private:
  DeviceHandle device_;
};

class HistogramKernelGPU : public VideoKernel {
public:
  HistogramKernelGPU(const Kernel::Config& config)
    : VideoKernel(config),
      device_(config.devices[0]),
      num_cuda_streams_(32),
      streams_(num_cuda_streams_)
    {}

  void new_frame_info() override {
    streams_.resize(0);
    streams_.resize(num_cuda_streams_);
    planes_.clear();
    for (i32 i = 0; i < 3; ++i) {
      planes_.push_back(
          cvc::GpuMat(frame_info_.height, frame_info_.width, CV_8UC1));
    }
  }

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    check_frame_info(device_, input_columns[1]);

    size_t hist_size = BINS * 3 * sizeof(float);
    i32 input_count = input_columns[0].rows.size();
    u8* output_block = new_block_buffer(
      device_, hist_size * input_count, input_count);

    for (i32 i = 0; i < input_count; ++i) {
      i32 sid = i % num_cuda_streams_;
      cv::cuda::Stream& s = streams_[sid];

      cvc::GpuMat img(frame_info_.height, frame_info_.width, CV_8UC3,
                      input_columns[0].rows[i].buffer);
      cvc::split(img, planes_, s);

      u8* output_buf = output_block + i * hist_size;
      cvc::GpuMat out_mat(1, BINS * 3, CV_32S, output_buf);

      for (i32 j = 0; j < 3; ++j) {
        cvc::histEven(planes_[j],
                      out_mat(cv::Rect(j * BINS, 0, BINS, 1)),
                      BINS,
                      0, 256,
                      s);
      }

      output_columns[0].rows.push_back(Row{output_buf, hist_size});
    }

    for (cv::cuda::Stream& s : streams_) {
      s.waitForCompletion();
    }
  }

  void set_device() {
#ifdef HAVE_CUDA
    CU_CHECK(cudaSetDevice(device_.id));
#else
    LOG(FATAL) << "Cuda not enabled.";
#endif
    cvc::setDevice(device_.id);
  }

private:
  DeviceHandle device_;
  i32 num_cuda_streams_;
  std::vector<cv::cuda::Stream> streams_;
  std::vector<cvc::GpuMat> planes_;
};

REGISTER_EVALUATOR(Histogram).outputs({"histogram"});

REGISTER_KERNEL(Histogram, HistogramKernelCPU)
    .device(DeviceType::CPU)
    .num_devices(1);

REGISTER_KERNEL(Histogram, HistogramKernelGPU)
    .device(DeviceType::GPU)
    .num_devices(1);

}
