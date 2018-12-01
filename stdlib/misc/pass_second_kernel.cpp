#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"

namespace scanner {

  class PassSecondKernel : public BatchedKernel {
  public:
    PassSecondKernel(const KernelConfig& config)
        : BatchedKernel(config),
          device_(config.devices[0]) {}

    void execute(const BatchedElements& input_columns,
                 BatchedElements& output_columns) override {
      for (auto& element : input_columns[0]) {
        add_buffer_ref(device_, element.buffer);
      }
      for (auto& element : input_columns[1]) {
        add_buffer_ref(device_, element.buffer);
      }
      output_columns[0] = input_columns[1];
    }

  private:
    DeviceHandle device_;
  };

  REGISTER_OP(PassSecond).input("data1").input("data2").output("output");

  REGISTER_KERNEL(PassSecond, PassSecondKernel)
      .device(DeviceType::CPU)
      .batch()
      .num_devices(1);

  REGISTER_KERNEL(PassSecond, PassSecondKernel)
      .device(DeviceType::GPU)
      .batch()
      .num_devices(1);

}
