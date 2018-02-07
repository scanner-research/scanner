#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"

namespace scanner {

// Dummy Kernel
class UnsliceKernel : public BatchedKernel {
 public:
  UnsliceKernel(const KernelConfig& config)
    : BatchedKernel(config) {}

  void execute(const BatchedElements& input_columns,
               BatchedElements& output_columns) override {
    // No implementation
  }
};


// Reserve Op name as builtin
REGISTER_OP(Unslice).input("col").output("out");

REGISTER_KERNEL(Unslice, UnsliceKernel).device(DeviceType::CPU).num_devices(1);

REGISTER_KERNEL(Unslice, UnsliceKernel).device(DeviceType::GPU).num_devices(1);


REGISTER_OP(UnsliceFrame).frame_input("col").frame_output("out");

REGISTER_KERNEL(UnsliceFrame, UnsliceKernel)
    .device(DeviceType::CPU)
    .batch()
    .num_devices(1);

REGISTER_KERNEL(UnsliceFrame, UnsliceKernel)
    .device(DeviceType::GPU)
    .batch()
    .num_devices(1);

}
