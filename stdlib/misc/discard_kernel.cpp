#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"

namespace scanner {

class DiscardKernel : public Kernel {
 public:
  DiscardKernel(const Kernel::Config& config)
      : Kernel(config),
        device_(config.devices[0]),
        work_item_size_(config.work_item_size) {}

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    i32 input_count = (i32)num_rows(input_columns[0]);
    u8* output_block = new_block_buffer(device_, 1, input_count);
    for (i32 i = 0; i < input_count; ++i) {
      insert_element(output_columns[0], output_block, 1);
    }
  }

 private:
  DeviceHandle device_;
  i32 work_item_size_;
};

REGISTER_OP(Discard).input("ignore").output("dummy");

REGISTER_OP(DiscardFrame).frame_input("ignore").output("dummy");

REGISTER_KERNEL(Discard, DiscardKernel).device(DeviceType::CPU).num_devices(1);

REGISTER_KERNEL(Discard, DiscardKernel).device(DeviceType::GPU).num_devices(1);

REGISTER_KERNEL(DiscardFrame, DiscardKernel)
  .device(DeviceType::CPU)
  .num_devices(1);

REGISTER_KERNEL(DiscardFrame, DiscardKernel)
  .device(DeviceType::GPU)
  .num_devices(1);
}
