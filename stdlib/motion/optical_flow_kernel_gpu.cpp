#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/cuda.h"
#include "scanner/util/cycle_timer.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"

#include <opencv2/video.hpp>

namespace scanner {

class OpticalFlowKernelGPU : public StenciledKernel, public VideoKernel {
 public:
  OpticalFlowKernelGPU(const KernelConfig& config)
    : StenciledKernel(config),
      device_(config.devices[0]),
      work_item_size_(config.work_item_size),
      num_cuda_streams_(2) {
    set_device();
    streams_.resize(num_cuda_streams_);
    for (i32 i = 0; i < num_cuda_streams_; ++i) {
      flow_finders_.push_back(
          cvc::FarnebackOpticalFlow::create(3, 0.5, false, 15, 3, 5, 1.2, 0));
    }
  }

  ~OpticalFlowKernelGPU() { set_device(); }

  void new_frame_info() override {
    set_device();
    grayscale_.resize(0);
    for (i32 i = 0; i < 2; ++i) {
      grayscale_.emplace_back(frame_info_.height(), frame_info_.width(),
                              CV_8UC1);
    }
  }

  void reset() override {
    set_device();
    initial_frame_ = cvc::GpuMat();
    flow_finders_.resize(0);
    for (i32 i = 0; i < num_cuda_streams_; ++i) {
      flow_finders_.push_back(
          cvc::FarnebackOpticalFlow::create(3, 0.5, false, 15, 3, 5, 1.2, 0));
    }
  }

  void execute(const StenciledColumns& input_columns,
               Columns& output_columns) override {
    auto& frame_col = input_columns[0];
    check_frame(device_, frame_col[0]);
    set_device();

    i32 input_count = (i32)num_rows(frame_col);
    FrameInfo out_frame_info(frame_info_.height(), frame_info_.width(), 2,
                             FrameType::F32);
    Frame* output_frame = new_frame(device_, out_frame_info);

    double start = CycleTimer::currentSeconds();

    cvc::GpuMat input0 =
        frame_to_gpu_mat(frame_col[0].as_const_frame());
    cvc::GpuMat input1 =
        frame_to_gpu_mat(frame_col[1].as_const_frame());
    cvc::cvtColor(input0, grayscale_[0], CV_BGR2GRAY, 0, streams_[0]);
    cvc::cvtColor(input1, grayscale_[1], CV_BGR2GRAY, 0, streams_[1]);

    for (cv::cuda::Stream& s : streams_) {
      s.waitForCompletion();
    }

    flow_finders_[0]->calc(grayscale_[0], grayscale_[1],
                           frame_to_gpu_mat(output_frame),
                           streams_[0]);

    for (cv::cuda::Stream& s : streams_) {
      s.waitForCompletion();
    }

    insert_frame(output_columns[0], output_frame);
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
