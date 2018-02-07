#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/cuda.h"
#include "scanner/util/cycle_timer.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"

#include <opencv2/video.hpp>

namespace scanner {

class OpticalFlowKernelGPU : public StenciledBatchedKernel, public VideoKernel {
 public:
  OpticalFlowKernelGPU(const KernelConfig& config)
    : StenciledBatchedKernel(config),
      device_(config.devices[0]),
      num_cuda_streams_(8) {
    set_device();
    cv::cuda::setBufferPoolUsage(true);
    cv::cuda::setBufferPoolConfig(device_.id, 50 * 1024 * 1024, 5);
    streams_.resize(num_cuda_streams_);
    for (i32 i = 0; i < num_cuda_streams_; ++i) {
      flow_finders_.push_back(
          cvc::FarnebackOpticalFlow::create(3, 0.5, false, 15, 3, 5, 1.2, 0));
    }
  }

  ~OpticalFlowKernelGPU() {
    set_device();
    flow_finders_.clear();
    streams_.clear();
    cv::cuda::setBufferPoolConfig(device_.id, 0, 0);
    cv::cuda::setBufferPoolUsage(false);
  }

  void new_frame_info() override {
    set_device();
  }

  void reset() override {
    set_device();
    initial_frame_ = cvc::GpuMat();
  }

  void execute(const StenciledBatchedElements& input_columns,
               BatchedElements& output_columns) override {
    set_device();

    auto& frame_col = input_columns[0];
    check_frame(device_, frame_col[0][0]);

    i32 input_count = (i32)frame_col.size();
    std::vector<const Frame*> input_frames;
    for (i32 i = 0; i < input_count; ++i) {
      input_frames.push_back(frame_col[i][0].as_const_frame());
    }
    input_frames.push_back(frame_col.back()[1].as_const_frame());

    grayscale_.resize(input_count + 1);

    FrameInfo out_frame_info(frame_info_.height(), frame_info_.width(), 2,
                             FrameType::F32);
    std::vector<Frame*> output_frames =
        new_frames(device_, out_frame_info, input_count);

    for (i32 i = 0; i < input_count + 1; ++i) {
      i32 sidx = i % num_cuda_streams_;
      streams_[sidx].waitForCompletion();
      cvc::GpuMat input = frame_to_gpu_mat(input_frames[i]);
      cvc::cvtColor(input, grayscale_[i], CV_BGR2GRAY, 0, streams_[sidx]);
    }
    for (auto& s : streams_) {
      s.waitForCompletion();
    }

    for (i32 i = 1; i < input_count + 1; ++i) {
      i32 sidx = i % num_cuda_streams_;

      i32 curr_idx = i;
      i32 prev_idx = (i - 1);

      cvc::GpuMat& input0 = grayscale_[curr_idx];
      cvc::GpuMat& input1 = grayscale_[prev_idx];

      //streams_[sidx].waitForCompletion();
      cvc::GpuMat output_mat = frame_to_gpu_mat(output_frames[i - 1]);
      flow_finders_[0]->calc(input0, input1, output_mat);
      insert_frame(output_columns[0], output_frames[i - 1]);
    }
    for (auto& s : streams_) {
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
  i32 num_cuda_streams_;
  std::vector<cv::cuda::Stream> streams_;
};

REGISTER_KERNEL(OpticalFlow, OpticalFlowKernelGPU)
    .device(DeviceType::GPU)
    .batch()
    .num_devices(1);
}
