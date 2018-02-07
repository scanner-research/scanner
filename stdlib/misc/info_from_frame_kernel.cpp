#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"

namespace scanner {

class InfoFromFrameKernel : public BatchedKernel {
 public:
  InfoFromFrameKernel(const KernelConfig& config)
    : BatchedKernel(config),
      device_(config.devices[0]) {}

  void execute(const BatchedElements& input_columns,
               BatchedElements& output_columns) override {
    i32 input_count = (i32)num_rows(input_columns[0]);
    u8* output_block =
        new_block_buffer(device_, sizeof(FrameInfo) * input_count, input_count);
    for (i32 i = 0; i < input_count; ++i) {
      const Frame* frame = input_columns[0][i].as_const_frame();

      u8* buffer = output_block + i * sizeof(FrameInfo);
      FrameInfo* info = reinterpret_cast<FrameInfo*>(buffer);
      FrameInfo info_cpu = frame->as_frame_info();
      memcpy_buffer((u8*) info, device_,
                    (u8*) &info_cpu, CPU_DEVICE,
                    sizeof(FrameInfo));
      insert_element(output_columns[0], buffer, sizeof(FrameInfo));
    }
  }

 private:
  DeviceHandle device_;
};

REGISTER_OP(InfoFromFrame).frame_input("frame").output("frame_info");

REGISTER_KERNEL(InfoFromFrame, InfoFromFrameKernel)
    .device(DeviceType::CPU)
    .num_devices(1);

REGISTER_KERNEL(InfoFromFrame, InfoFromFrameKernel)
    .device(DeviceType::GPU)
    .num_devices(1);
}
