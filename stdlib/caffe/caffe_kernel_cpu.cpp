#include "stdlib/caffe/caffe_kernel.h"

namespace scanner {

REGISTER_OP(Caffe)
    .inputs({"caffe_frame", "frame_info"})
    .outputs({"caffe_output"});
REGISTER_KERNEL(Caffe, CaffeKernel).device(DeviceType::CPU).num_devices(1);
}
