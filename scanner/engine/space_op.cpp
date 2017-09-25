#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"

namespace scanner {

// Dummy Kernel
class SpaceKernel : public Kernel {
 public:
  SpaceKernel(const KernelConfig& config)
    : Kernel(config),
      device_(config.devices[0]) {}

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    // No implementation
  }
};


// Reserve Op name as builtin
REGISTER_OP(Space).input("in").output("out");

REGISTER_KERNEL(Space, SpaceKernel).device(DeviceType::CPU).num_devices(1);

REGISTER_KERNEL(Space, SpaceKernel).device(DeviceType::GPU).num_devices(1);


REGISTER_OP(SpaceFrame).frame_input("in").frame_output("out");

REGISTER_KERNEL(DiscardFrame, DiscardKernel)
    .device(DeviceType::CPU)
    .batch()
    .num_devices(1);

REGISTER_KERNEL(DiscardFrame, DiscardKernel)
    .device(DeviceType::GPU)
    .batch()
    .num_devices(1);

}
