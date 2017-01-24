#pragma once

#include "scanner/kernels/args.pb.h"
#include "scanner/api/evaluator.h"
#include "scanner/api/kernel.h"
#include "scanner/util/cuda.h"
#include "scanner/util/memory.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

namespace scanner {

class CaffeKernel : public VideoKernel {
public:
  CaffeKernel(const Kernel::Config& config);
  void new_frame_info() override;
  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override;
  void set_device();

protected:
  DeviceHandle device_;
  proto::CaffeArgs args_;
  std::unique_ptr<caffe::Net<float>> net_;
};

proto::NetDescriptor descriptor_from_net_file(const std::string& path);

template<typename T>
std::vector<T> repeatedptrfield_to_vector(
  const google::protobuf::RepeatedPtrField<T> ptrfield) {
  std::vector<T> vec;
  for (auto& el : ptrfield) {
    vec.push_back(el);
  }
  return vec;
}

#define REGISTER_CAFFE_EVALUATOR(net__, desc__)       \
  REGISTER_CAFFE_EVALUATOR_HELPER(__COUNTER__, net__, desc__)

#define REGISTER_CAFFE_EVALUATOR_HELPER(uid__, net__, desc__) \
  REGISTER_CAFFE_EVALUATOR_UID(uid__, net__, desc__)

#define REGISTER_CAFFE_EVALUATOR_UID(uid__, net__, desc__)    \
  static ::scanner::proto::NetDescriptor net_descriptor_##uid__ =       \
    descriptor_from_net_file(desc__);                                   \
  REGISTER_EVALUATOR(net__).outputs(                                    \
    repeatedptrfield_to_vector(net_descriptor_##uid__.output_layer_names())); \

#define REGISTER_CAFFE_KERNELS(net__, kernel__)                                  \
  REGISTER_KERNEL(net__, kernel__).device(DeviceType::CPU).num_devices(1); \
  REGISTER_KERNEL(net__, kernel__).device(DeviceType::GPU).num_devices(1);

}
