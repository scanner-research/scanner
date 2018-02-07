#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"

namespace scanner {

// Dummy Kernel
class SpaceKernel : public BatchedKernel {
 public:
  SpaceKernel(const KernelConfig& config)
    : BatchedKernel(config) {}

  void execute(const BatchedElements& input_columns,
               BatchedElements& output_columns) override {
    // No implementation
  }
};


// Reserve Op name as builtin
REGISTER_OP(Space).input("col").output("out");

REGISTER_KERNEL(Space, SpaceKernel)
    .device(DeviceType::CPU)
    .batch()
    .num_devices(1);

REGISTER_KERNEL(Space, SpaceKernel)
    .device(DeviceType::GPU)
    .batch()
    .num_devices(1);

REGISTER_OP(SpaceFrame).frame_input("col").frame_output("out");

REGISTER_KERNEL(spaceFrame, SpaceKernel)
    .device(DeviceType::CPU)
    .batch()
    .num_devices(1);

REGISTER_KERNEL(SpaceFrame, SpaceKernel)
    .device(DeviceType::GPU)
    .batch()
    .num_devices(1);
}
