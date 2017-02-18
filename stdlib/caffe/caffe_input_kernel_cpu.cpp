#include "stdlib/caffe/caffe_input_kernel.h"

namespace scanner {

REGISTER_OP(CaffeInput)
    .inputs({"frame", "frame_info"})
    .outputs({"caffe_frame"});

REGISTER_KERNEL(CaffeInput, CaffeInputKernel)
    .device(DeviceType::CPU)
    .num_devices(1);
}
