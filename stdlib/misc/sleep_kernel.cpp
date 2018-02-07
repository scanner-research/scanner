#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"

namespace scanner {

class SleepKernel : public Kernel {
 public:
  SleepKernel(const KernelConfig& config)
    : Kernel(config),
      device_(config.devices[0]) {}

  void execute(const Elements& input_columns, Elements& output_columns) override {
    sleep(2);
    insert_element(output_columns[0], new_buffer(device_, 1), 1);
  }

 private:
  DeviceHandle device_;
};

REGISTER_OP(Sleep).input("ignore").output("dummy");

REGISTER_OP(SleepFrame).frame_input("ignore").output("dummy");

REGISTER_KERNEL(Sleep, SleepKernel).device(DeviceType::CPU).num_devices(1);

REGISTER_KERNEL(Sleep, SleepKernel).device(DeviceType::GPU).num_devices(1);

REGISTER_KERNEL(SleepFrame, SleepKernel).device(DeviceType::CPU).num_devices(1);

REGISTER_KERNEL(SleepFrame, SleepKernel).device(DeviceType::GPU).num_devices(1);
}
