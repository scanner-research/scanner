#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/cuda.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"
#include "stdlib/stdlib.pb.h"

#ifdef HAVE_CUDA
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif

namespace scanner {

class FacenetInputKernel : public VideoKernel {
 public:
  FacenetInputKernel(const Kernel::Config& config)
      : VideoKernel(config),
        device_(config.devices[0])
#ifdef HAVE_CUDA
        ,
        num_cuda_streams_(32),
        streams_(num_cuda_streams_)
#endif
  {
    proto::FacenetArgs args;
    args.ParseFromArray(config.args.data(), config.args.size());
    args_.CopyFrom(args.caffe_args());
    scale_ = args.scale();
  }

  void new_frame_info() override {
    net_input_width_ = std::floor(frame_info_.shape[1] * scale_);
    net_input_height_ = std::floor(frame_info_.shape[2] * scale_);
    if (net_input_width_ % 8 != 0) {
      net_input_width_ += 8 - (net_input_width_ % 8);
    };
    if (net_input_height_ % 8 != 0) {
      net_input_height_ += 8 - (net_input_height_ % 8);
    }

    cv::cuda::setDevice(device_.id);
    cudaSetDevice(device_.id);

    mean_mat_g_ =
        cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_32FC3,
                         cv::Scalar(args_.net_descriptor().mean_colors(0),
                                    args_.net_descriptor().mean_colors(1),
                                    args_.net_descriptor().mean_colors(2)));

    frame_input_g_.clear();
    resized_input_g_.clear();
    float_input_g_.clear();
    flipped_planes_g_.clear();
    normalized_input_g_.clear();
    input_planes_g_.clear();
    planar_input_g_.clear();
    flipped_planes_g_.clear();
    for (size_t i = 0; i < num_cuda_streams_; ++i) {
      frame_input_g_.push_back(
          cv::cuda::GpuMat(frame_info_.shape[2], frame_info_.shape[1], CV_8UC3));
      resized_input_g_.push_back(
          cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_8UC3));
      float_input_g_.push_back(
          cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_32FC3));
      normalized_input_g_.push_back(
          cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_32FC3));
      std::vector<cv::cuda::GpuMat> planes;
      std::vector<cv::cuda::GpuMat> flipped_planes;
      for (i32 i = 0; i < 3; ++i) {
        planes.push_back(
            cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_32FC1));
        flipped_planes.push_back(
            cv::cuda::GpuMat(net_input_width_, net_input_height_, CV_32FC1));
      }
      input_planes_g_.push_back(planes);
      flipped_planes_g_.push_back(flipped_planes);
      planar_input_g_.push_back(
          cv::cuda::GpuMat(net_input_width_ * 3, net_input_height_, CV_32FC1));
    }
  }

  // TODO(wcrichto): set_device

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    auto& frame_col = input_columns[0];
    check_frame(device_, frame_col[0]);

    i32 input_count = (i32)frame_col.size();
    FrameInfo net_input_info(net_input_width_, net_input_height_, 3,
                             FrameType::F32);
    i32 net_input_size = net_input_info.size();
    std::vector<Frame*> output_frames =
        new_frames(device_, net_input_info, input_count);

    streams_.resize(0);
    streams_.resize(num_cuda_streams_);

    for (i32 i = 0; i < input_count; ++i) {
      int sid = i % num_cuda_streams_;
      cv::cuda::Stream& cv_stream = streams_[sid];

      Frame* output_frame = output_frames[i];
      f32* net_input = (f32*)output_frame->data;

      // Convert input frame to gpu mat
      frame_input_g_[sid] = frame_to_gpu_mat(frame_col[i].as_const_frame());

      cv::cuda::resize(frame_input_g_[sid], resized_input_g_[sid],
                       cv::Size(net_input_width_, net_input_height_), 0, 0,
                       cv::INTER_LINEAR, cv_stream);
      resized_input_g_[sid].convertTo(float_input_g_[sid], CV_32FC3, cv_stream);
      cv::cuda::subtract(float_input_g_[sid], mean_mat_g_,
                         normalized_input_g_[sid], cv::noArray(), -1,
                         cv_stream);
      // Changed from interleaved RGB to planar RGB
      cv::cuda::split(normalized_input_g_[sid], input_planes_g_[sid],
                      cv_stream);
      cv::cuda::transpose(input_planes_g_[sid][0], flipped_planes_g_[sid][0],
                          cv_stream);
      cv::cuda::transpose(input_planes_g_[sid][1], flipped_planes_g_[sid][1],
                          cv_stream);
      cv::cuda::transpose(input_planes_g_[sid][2], flipped_planes_g_[sid][2],
                          cv_stream);
      auto& plane1 = flipped_planes_g_[sid][0];
      auto& plane2 = flipped_planes_g_[sid][1];
      auto& plane3 = flipped_planes_g_[sid][2];
      auto& planar_input = planar_input_g_[sid];
      plane1.copyTo(planar_input(cv::Rect(
          0, net_input_width_ * 0, net_input_height_, net_input_width_)));
      plane2.copyTo(planar_input(cv::Rect(
          0, net_input_width_ * 1, net_input_height_, net_input_width_)));
      plane3.copyTo(planar_input(cv::Rect(
          0, net_input_width_ * 2, net_input_height_, net_input_width_)));
      assert(planar_input.cols == net_input_height_);
      cudaStream_t s = cv::cuda::StreamAccessor::getStream(cv_stream);
      CU_CHECK(cudaMemcpy2DAsync(
          net_input, net_input_height_ * sizeof(float), planar_input.data,
          planar_input.step, net_input_height_ * sizeof(float),
          net_input_width_ * 3, cudaMemcpyDeviceToDevice, s));

      INSERT_FRAME(output_columns[0], output_frame);
    }
    for (cv::cuda::Stream& s : streams_) {
      s.waitForCompletion();
    }
  }

 private:
  DeviceHandle device_;
  proto::CaffeArgs args_;
  f32 scale_;
  i32 net_input_width_;
  i32 net_input_height_;

  i32 num_cuda_streams_;
  cv::cuda::GpuMat mean_mat_g_;
  std::vector<cv::cuda::Stream> streams_;
  std::vector<cv::cuda::GpuMat> frame_input_g_;
  std::vector<cv::cuda::GpuMat> resized_input_g_;
  std::vector<cv::cuda::GpuMat> float_input_g_;
  std::vector<cv::cuda::GpuMat> normalized_input_g_;
  std::vector<std::vector<cv::cuda::GpuMat>> input_planes_g_;
  std::vector<std::vector<cv::cuda::GpuMat>> flipped_planes_g_;
  std::vector<cv::cuda::GpuMat> planar_input_g_;
};

REGISTER_OP(FacenetInput)
    .frame_input("frame")
    .frame_output("facenet_input");

REGISTER_KERNEL(FacenetInput, FacenetInputKernel)
    .device(DeviceType::GPU)
    .num_devices(1);
}
