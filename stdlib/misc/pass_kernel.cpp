#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"

namespace scanner {

class PassKernel : public BatchedKernel {
 public:
  PassKernel(const KernelConfig& config)
    : BatchedKernel(config),
      device_(config.devices[0]) {}

  void execute(const BatchedElements& input_columns,
               BatchedElements& output_columns) override {
    for (auto& element : input_columns[0]) {
      add_buffer_ref(device_, element.buffer);
    }
    output_columns[0] = input_columns[0];
  }

 private:
  DeviceHandle device_;
};

REGISTER_OP(Pass).input("input").output("output");

REGISTER_KERNEL(Pass, PassKernel)
    .device(DeviceType::CPU)
    .batch()
    .num_devices(1);

REGISTER_KERNEL(Pass, PassKernel)
    .device(DeviceType::GPU)
    .batch()
    .num_devices(1);

}
