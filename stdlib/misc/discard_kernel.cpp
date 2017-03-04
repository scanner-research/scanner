#include "scanner/api/op.h"
#include "scanner/api/kernel.h"
#include "scanner/util/memory.h"

namespace scanner {

class DiscardKernel : public Kernel {
public:
  DiscardKernel(const Kernel::Config &config)
      : Kernel(config), device_(config.devices[0]),
        work_item_size_(config.work_item_size) {}

  void execute(const BatchedColumns &input_columns,
               BatchedColumns &output_columns) override {
    i32 input_count = (i32)input_columns[0].rows.size();
    u8 *output_block = new_block_buffer(device_, 1, input_count);
    for (i32 i = 0; i < input_count; ++i) {
      output_columns[0].rows.push_back(Row{output_block, 1});
    }
  }

private:
  DeviceHandle device_;
  i32 work_item_size_;
};

REGISTER_OP(Discard).inputs({"ignore"}).outputs({"dummy"});

REGISTER_KERNEL(Discard, DiscardKernel)
    .device(DeviceType::CPU)
    .num_devices(1);

REGISTER_KERNEL(Discard, DiscardKernel)
    .device(DeviceType::GPU)
    .num_devices(1);
}
