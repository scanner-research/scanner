#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"

namespace scanner {

class DiscardKernel : public BatchedKernel {
 public:
  DiscardKernel(const KernelConfig& config)
    : BatchedKernel(config),
      device_(config.devices[0]),
      work_item_size_(config.work_item_size) {}

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    i32 input_count = (i32)num_rows(input_columns[0]);
    for (i32 i = 0; i < input_count; ++i) {
      insert_element(output_columns[0], new_buffer(device_, 1), 1);
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
    .batch()
    .num_devices(1);

REGISTER_KERNEL(DiscardFrame, DiscardKernel)
    .device(DeviceType::GPU)
    .batch()
    .num_devices(1);
}
