#include "scanner/api/op.h"
#include "kernels/caffe_kernel.h"

namespace scanner {

class FacenetKernel : public CaffeKernel {
public:
  FacenetKernel(const Kernel::Config& config)
    : CaffeKernel(get_caffe_config(config)) {}

  void net_config() override {
    // Calculate width by scaling by box size
    int resize_width = frame_info_.width() * scale_;
    int resize_height = frame_info_.height() * scale_;

    resize_width += (resize_width % 8);
    resize_height += (resize_height % 8);

    int net_input_width = resize_height;
    int net_input_height = resize_width;

    const boost::shared_ptr<caffe::Blob<float>> input_blob{
      net_->blob_by_name("data")};
    input_blob->Reshape({input_blob->shape(0), input_blob->shape(1),
          net_input_height, net_input_width});
  }

  Kernel::Config get_caffe_config(const Kernel::Config& config) {
    proto::FacenetArgs args;
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

REGISTER_OP(Facenet).outputs({"caffe_output"});
REGISTER_KERNEL(Facenet, FacenetKernel).device(DeviceType::CPU).num_devices(1);
REGISTER_KERNEL(Facenet, FacenetKernel).device(DeviceType::GPU).num_devices(1);

}
