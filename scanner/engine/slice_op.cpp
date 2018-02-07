#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"

namespace scanner {

// Dummy Kernel
class SliceKernel : public BatchedKernel {
 public:
  SliceKernel(const KernelConfig& config)
    : BatchedKernel(config) {}

  void execute(const BatchedElements& input_columns,
               BatchedElements& output_columns) override {
    // No implementation
  }
};


// Reserve Op name as builtin
REGISTER_OP(Slice).input("col").output("out");

REGISTER_KERNEL(Slice, SliceKernel).device(DeviceType::CPU).num_devices(1);

REGISTER_KERNEL(Slice, SliceKernel).device(DeviceType::GPU).num_devices(1);


REGISTER_OP(SliceFrame).frame_input("col").frame_output("out");

REGISTER_KERNEL(SliceFrame, SliceKernel)
    .device(DeviceType::CPU)
    .batch()
    .num_devices(1);

REGISTER_KERNEL(SliceFrame, SliceKernel)
    .device(DeviceType::GPU)
    .batch()
    .num_devices(1);

}
