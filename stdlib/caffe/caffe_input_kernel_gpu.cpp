#include "stdlib/caffe/caffe_input_kernel.h"

namespace scanner {

REGISTER_KERNEL(CaffeInput, CaffeInputKernel)
    .device(DeviceType::GPU)
    .num_devices(1);
}
