#pragma once

#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/cuda.h"
#include "scanner/util/opencv.h"
#include "stdlib/stdlib.pb.h"

#ifdef HAVE_CUDA
#include "caffe_input_transformer_gpu/caffe_input_transformer_gpu.h"
#endif
#include "caffe_input_transformer_cpu/caffe_input_transformer_cpu.h"

namespace scanner {

class CaffeInputKernel : public VideoKernel {
 public:
  CaffeInputKernel(const Kernel::Config& config);
  ~CaffeInputKernel();

  void new_frame_info() override;

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override;

  void set_device();

  virtual void extra_inputs(const BatchedColumns& input_columns,
                            BatchedColumns& output_columns) {}

 protected:
  void set_halide_buf(buffer_t& halide_buf, u8* buf, size_t size);
  void unset_halide_buf(buffer_t& halide_buf);
  void transform_halide(const u8* input_buffer, u8* output_buffer);
  void transform_caffe(u8* input_buffer, u8* output_buffer);

  DeviceHandle device_;
  proto::CaffeInputArgs args_;
  i32 net_input_width_;
  i32 net_input_height_;
#ifdef HAVE_CUDA
  CUcontext context_;
#endif
};
}
