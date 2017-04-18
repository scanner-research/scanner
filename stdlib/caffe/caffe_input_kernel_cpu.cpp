#include "stdlib/caffe/caffe_input_kernel.h"

namespace scanner {

REGISTER_OP(CaffeInput)
    .frame_input("frame")
    .frame_output("caffe_frame");

REGISTER_KERNEL(CaffeInput, CaffeInputKernel)
    .device(DeviceType::CPU)
    .num_devices(1);
}
