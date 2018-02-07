#pragma once

#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/cuda.h"
#include "scanner/util/memory.h"
#include "stdlib/stdlib.pb.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

namespace scanner {

using CustomNetConfiguration = void (*)(const FrameInfo& frame_info,
                                        caffe::Net<float>* net);

class CaffeKernel : public BatchedKernel, public VideoKernel {
 public:
  CaffeKernel(const KernelConfig& config);
  void validate(proto::Result* result) override;
  void new_frame_info() override;
  void execute(const BatchedElements& input_columns,
               BatchedElements& output_columns) override;
  void set_device();

  virtual void net_config() {}

 protected:
  proto::Result valid_;
  DeviceHandle device_;
  proto::CaffeArgs args_;
  std::unique_ptr<caffe::Net<float>> net_;
  CustomNetConfiguration net_config_;
};

proto::NetDescriptor descriptor_from_net_file(const std::string& path);
}
