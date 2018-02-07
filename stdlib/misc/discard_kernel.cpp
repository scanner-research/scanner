#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"

namespace scanner {

class DiscardKernel : public BatchedKernel {
 public:
  DiscardKernel(const KernelConfig& config)
    : BatchedKernel(config),
      device_(config.devices[0]) {}

  void execute(const BatchedElements& input_columns,
               BatchedElements& output_columns) override {
    i32 input_count = (i32)num_rows(input_columns[0]);
    u8* output_block = new_block_buffer(device_, input_count, input_count);
    for (i32 i = 0; i < input_count; ++i) {
      insert_element(output_columns[0], output_block + i, 1);
    }
  }

 private:
  DeviceHandle device_;
};

REGISTER_OP(Discard).input("ignore").output("dummy");

REGISTER_OP(DiscardFrame).frame_input("ignore").output("dummy");

REGISTER_KERNEL(Discard, DiscardKernel).device(DeviceType::CPU).batch().num_devices(1);

REGISTER_KERNEL(Discard, DiscardKernel).device(DeviceType::GPU).batch().num_devices(1);

REGISTER_KERNEL(DiscardFrame, DiscardKernel)
    .device(DeviceType::CPU)
    .batch()
    .num_devices(1);

REGISTER_KERNEL(DiscardFrame, DiscardKernel)
    .device(DeviceType::GPU)
    .batch()
    .num_devices(1);
}
