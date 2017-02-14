#include "stdlib/caffe_input_kernel.h"

namespace scanner {

REGISTER_OP(CaffeInput).outputs({"caffe_frame"});

REGISTER_KERNEL(CaffeInput, CaffeInputKernel)
    .device(DeviceType::CPU)
    .num_devices(1);
}
