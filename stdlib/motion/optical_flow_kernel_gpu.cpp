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
      work_item_size_(config.work_item_size),
      num_cuda_streams_(4) {
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
    flow_finders_.clear();
    for (i32 i = 0; i < num_cuda_streams_; ++i) {
      flow_finders_.push_back(
          cvc::FarnebackOpticalFlow::create(3, 0.5, false, 15, 3, 5, 1.2, 0));
    }
  }

  void execute(const StenciledBatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    set_device();

    auto& frame_col = input_columns[0];
    check_frame(device_, frame_col[0][0]);

    i32 input_count = (i32)frame_col.size();
    std::vector<const Frame*> input_frames;
    for (i32 i = 0; i < input_count; ++i) {
      input_frames.push_back(frame_col[i][0].as_const_frame());
    }
    input_frames.push_back(frame_col.back()[1].as_const_frame());

    FrameInfo out_frame_info(frame_info_.height(), frame_info_.width(), 2,
                             FrameType::F32);
    std::vector<Frame*> output_frames =
        new_frames(device_, out_frame_info, input_count);

    cvc::GpuMat input0 = frame_to_gpu_mat(input_frames[0]);
    cvc::cvtColor(input0, grayscale_[0], CV_BGR2GRAY, 0, streams_[0]);
    cvc::Event prev_gray;
    prev_gray.record(streams_[0]);
    for (i32 i = 1; i < input_count + 1; ++i) {
      i32 curr_idx = i % 2;
      i32 prev_idx = (i - 1) % 2;

      cvc::GpuMat input1 = frame_to_gpu_mat(input_frames[i]);
      cvc::cvtColor(input1, grayscale_[curr_idx], CV_BGR2GRAY, 0,
                    streams_[curr_idx]);

      cvc::Event curr_gray;
      curr_gray.record(streams_[curr_idx]);
      streams_[curr_idx].waitEvent(prev_gray);
      prev_gray = curr_gray;
      cvc::GpuMat output_mat = frame_to_gpu_mat(output_frames[i - 1]);
      flow_finders_[curr_idx]->calc(grayscale_[prev_idx], grayscale_[curr_idx],
                                    output_mat, streams_[curr_idx]);

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
  i32 work_item_size_;
  i32 num_cuda_streams_;
  std::vector<cv::cuda::Stream> streams_;
};

REGISTER_KERNEL(OpticalFlow, OpticalFlowKernelGPU)
    .device(DeviceType::GPU)
    .batch()
    .num_devices(1);
}
