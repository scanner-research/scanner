#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"

namespace scanner {

class TestIncrementKernel : public Kernel {
 public:
  TestIncrementKernel(const KernelConfig& config)
    : Kernel(config),
      device_(config.devices[0]) {}

  void reset() {
    next_int_ = 0;
  }

  void execute(const Elements& input_columns,
               Elements& output_columns) override {
    if (last_row_ + 1 != input_columns[0].index) {
      last_row_ = input_columns[0].index - 1;
      reset();
    }
    last_row_++;

    u8* buffer = new_buffer(device_, sizeof(i64));
    *((i64*)buffer) = next_int_++;
    insert_element(output_columns[0], buffer, sizeof(i64));
  }

 private:
  DeviceHandle device_;
  i64 next_int_ = 0;
  i64 last_row_ = 0;
};

REGISTER_OP(TestIncrementUnbounded)
.input("ignore")
.output("integer")
.unbounded_state();

REGISTER_OP(TestIncrementUnboundedFrame)
.frame_input("ignore")
.output("integer")
.unbounded_state();

REGISTER_KERNEL(TestIncrementUnbounded, TestIncrementKernel)
.device(DeviceType::CPU)
.num_devices(1);

REGISTER_KERNEL(TestIncrementUnboundedFrame, TestIncrementKernel)
.device(DeviceType::CPU)
.num_devices(1);

REGISTER_OP(TestIncrementBounded)
.input("ignore")
.output("integer")
.bounded_state();

REGISTER_OP(TestIncrementBoundedFrame)
.frame_input("ignore")
.output("integer")
.bounded_state();

REGISTER_KERNEL(TestIncrementBounded, TestIncrementKernel)
.device(DeviceType::CPU)
.num_devices(1);

REGISTER_KERNEL(TestIncrementBoundedFrame, TestIncrementKernel)
.device(DeviceType::CPU)
.num_devices(1);
}
