#include "stdlib/caffe/caffe_kernel.h"

namespace scanner {

REGISTER_KERNEL(Caffe, CaffeKernel)
    .device(DeviceType::GPU)
    .batch()
    .num_devices(1);
}
