#include "scanner/api/op.h"
#include "stdlib/caffe/caffe_kernel.h"

namespace scanner {

class FasterRCNNKernel : public CaffeKernel {
 public:
  FasterRCNNKernel(const Kernel::Config& config) : CaffeKernel(config) {}

  void net_config() override {
    boost::shared_ptr<caffe::Blob<float>> blob = net_->blob_by_name("im_info");
    f32 buf[3] = {frame_info_.shape[2], frame_info_.shape[1], 1.0};
    f32* blob_data = device_.type == DeviceType::GPU ? blob->mutable_gpu_data()
                                                     : blob->mutable_cpu_data();
    memcpy_buffer((u8*)blob_data, device_, (u8*)buf, CPU_DEVICE,
                  3 * sizeof(f32));
  }
};

REGISTER_OP(FasterRCNN)
    .frame_input("caffe_input")
    .frame_output("cls_prob")
    .frame_output("rois")
    .frame_output("fc7");

REGISTER_KERNEL(FasterRCNN, FasterRCNNKernel)
    .device(DeviceType::CPU)
    .num_devices(1);
REGISTER_KERNEL(FasterRCNN, FasterRCNNKernel)
    .device(DeviceType::GPU)
    .num_devices(1);
}
