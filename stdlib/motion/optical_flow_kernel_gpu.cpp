#include "scanner/api/op.h"
#include "scanner/api/kernel.h"
#include "scanner/util/cuda.h"
#include "scanner/util/cycle_timer.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"

#include <opencv2/video.hpp>

namespace scanner {

class OpticalFlowKernelGPU : public VideoKernel {
public:
  OpticalFlowKernelGPU(const Kernel::Config &config)
      : VideoKernel(config), device_(config.devices[0]),
        work_item_size_(config.work_item_size), num_cuda_streams_(4) {
    set_device();
    streams_.resize(num_cuda_streams_);
    for (i32 i = 0; i < num_cuda_streams_; ++i) {
      flow_finders_.push_back(cvc::FarnebackOpticalFlow::create());
    }
  }

  ~OpticalFlowKernelGPU() {
    set_device();
  }

  void new_frame_info() override {
    set_device();
    grayscale_.resize(0);
    for (i32 i = 0; i < work_item_size_; ++i) {
      grayscale_.emplace_back(frame_info_.height(), frame_info_.width(),
                              CV_8UC1);
    }
  }

  void reset() override {
    set_device();
    initial_frame_ = cvc::GpuMat();
    flow_finders_.resize(0);
    for (i32 i = 0; i < num_cuda_streams_; ++i) {
      flow_finders_.push_back(cvc::FarnebackOpticalFlow::create());
    }
  }

  void execute(const BatchedColumns &input_columns,
               BatchedColumns &output_columns) override {
    auto& frame_col = input_columns[0];
    auto& frame_info_col = input_columns[1];
    check_frame_info(device_, frame_info_col);
    set_device();


    i32 input_count = (i32)frame_col.rows.size();
    size_t out_buf_size =
        frame_info_.width() * frame_info_.height() * 2 * sizeof(float);

    u8 *output_block =
        new_block_buffer(device_, out_buf_size * input_count, input_count);

    for (i32 i = 0; i < input_count; ++i) {
      i32 sid = i % num_cuda_streams_;
      cv::cuda::Stream &s = streams_[sid];
      cvc::GpuMat input(frame_info_.height(), frame_info_.width(), CV_8UC3,
                        frame_col.rows[i].buffer);
      cvc::cvtColor(input, grayscale_[i], CV_BGR2GRAY, 0, s);
    }

    for (cv::cuda::Stream &s : streams_) {
      s.waitForCompletion();
    }

    double start = CycleTimer::currentSeconds();

    // FIXME(wcrichto): TVL1 flow doesn't seem to work with multiple Cuda
    // streams, segfaults in the TVL1 destructor. Investigate?
    for (i32 i = 0; i < input_count; ++i) {
      i32 sid = i % num_cuda_streams_;
      cv::cuda::Stream &s = streams_[sid];
      cvc::GpuMat flow(frame_info_.height(), frame_info_.width(), CV_32FC2,
                       output_block + i * out_buf_size);

      if (i == 0) {
        if (initial_frame_.empty()) {
          output_columns[0].rows.push_back(Row{flow.data, out_buf_size});
          continue;
        } else {
          flow_finders_[sid]->calc(initial_frame_, grayscale_[0], flow, s);
        }
      } else {
        flow_finders_[sid]->calc(grayscale_[i - 1], grayscale_[i], flow, s);
      }

      output_columns[0].rows.push_back(Row{flow.data, out_buf_size});
    }

    grayscale_[input_count - 1].copyTo(initial_frame_);

    for (cv::cuda::Stream &s : streams_) {
      s.waitForCompletion();
    }
  }

private:
  void set_device() {
    CU_CHECK(cudaSetDevice(device_.id));
    cvc::setDevice(device_.id);
  }

  DeviceHandle device_;
  std::vector<cv::Ptr<cvc::DenseOpticalFlow>> flow_finders_;
  cvc::GpuMat initial_frame_;
  std::vector<cvc::GpuMat> grayscale_;
  i32 work_item_size_;
  i32 num_cuda_streams_;
  std::vector<cv::cuda::Stream> streams_;
};

REGISTER_KERNEL(OpticalFlow, OpticalFlowKernelGPU)
    .device(DeviceType::GPU)
    .num_devices(1);
}
