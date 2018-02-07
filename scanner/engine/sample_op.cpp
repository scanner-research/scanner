#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"

namespace scanner {

// Dummy Kernel
class SampleKernel : public BatchedKernel {
 public:
  SampleKernel(const KernelConfig& config)
    : BatchedKernel(config) {}

  void execute(const BatchedElements& input_columns,
               BatchedElements& output_columns) override {
    // No implementation
  }
};


// Reserve Op name as builtin
REGISTER_OP(Sample).input("col").output("out");

REGISTER_KERNEL(Sample, SampleKernel).device(DeviceType::CPU).num_devices(1);

REGISTER_KERNEL(Sample, SampleKernel).device(DeviceType::GPU).num_devices(1);


REGISTER_OP(SampleFrame).frame_input("col").frame_output("out");

REGISTER_KERNEL(SampleFrame, SampleKernel)
    .device(DeviceType::CPU)
    .batch()
    .num_devices(1);

REGISTER_KERNEL(SampleFrame, SampleKernel)
    .device(DeviceType::GPU)
    .batch()
    .num_devices(1);

}
