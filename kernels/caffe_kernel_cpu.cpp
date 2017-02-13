#include "kernels/caffe_kernel.h"

namespace scanner {

REGISTER_OP(Caffe).outputs({"caffe_output"});
REGISTER_KERNEL(Caffe, CaffeKernel).device(DeviceType::CPU).num_devices(1);

}
