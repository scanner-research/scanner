#include "scanner/api/op.h"
#include "stdlib/caffe/caffe_kernel.h"

#include "caffe/layers/imresize_layer.hpp"

namespace scanner {

class CPM2Kernel : public CaffeKernel {
 public:
  CPM2Kernel(const Kernel::Config& config)
      : CaffeKernel(get_caffe_config(config)) {}

  void net_config() override {
    int net_input_width = frame_info_.shape[0];
    int net_input_height = frame_info_.shape[1];

    caffe::ImResizeLayer<float>* resize_layer =
        (caffe::ImResizeLayer<float>*)net_->layer_by_name("resize").get();

    resize_layer->SetStartScale(1);
    resize_layer->SetScaleGap(0.1);
    resize_layer->setTargetDimenions(net_input_width, net_input_height);

    const boost::shared_ptr<caffe::Blob<float>> input_blob{
        net_->blob_by_name("image")};
    input_blob->Reshape({input_blob->shape(0), input_blob->shape(1),
                         net_input_height, net_input_width});
  }

  Kernel::Config get_caffe_config(const Kernel::Config& config) {
    proto::CPM2Args args;
    args.ParseFromArray(config.args.data(), config.args.size());
    scale_ = args.scale();

    Kernel::Config new_config(config);
    std::string caffe_string;
    args.caffe_args().SerializeToString(&caffe_string);
    new_config.args = std::vector<u8>(caffe_string.begin(), caffe_string.end());
    return new_config;
  }

 private:
  f32 scale_;
};

REGISTER_OP(CPM2)
    .frame_input("cpm2_input")
    .frame_output("cpm2_resized_map")
    .frame_output("cpm2_joints");

REGISTER_KERNEL(CPM2, CPM2Kernel).device(DeviceType::CPU).num_devices(1);
REGISTER_KERNEL(CPM2, CPM2Kernel).device(DeviceType::GPU).num_devices(1);
}
