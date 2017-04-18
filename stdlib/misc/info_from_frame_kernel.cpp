#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"

namespace scanner {

class InfoFromFrameKernel : public Kernel {
 public:
  InfoFromFrameKernel(const Kernel::Config& config)
      : Kernel(config),
        device_(config.devices[0]),
        work_item_size_(config.work_item_size) {}

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    i32 input_count = (i32)NUM_ROWS(input_columns);
    u8* output_block =
      new_block_buffer(device_, sizeof(FrameInfo) * input_count, input_count);
    for (i32 i = 0; i < input_count; ++i) {
      const Frame* frame = input_columns[0][i].as_const_frame();

      u8* buffer = output_block + i * sizeof(FrameInfo);
      FrameInfo* info = reinterpret_cast<FrameInfo*>(buffer);
      *info = frame->as_frame_info();
      INSERT_ELEMENT(output_columns[0], buffer, sizeof(FrameInfo));
    }
  }

 private:
  DeviceHandle device_;
  i32 work_item_size_;
};

REGISTER_OP(InfoFromFrame).frame_input("frame").output("frame_info");

REGISTER_KERNEL(InfoFromFrame, InfoFromFrameKernel)
  .device(DeviceType::CPU)
  .num_devices(1);

REGISTER_KERNEL(InfoFromFrame, InfoFromFrameKernel)
  .device(DeviceType::GPU)
  .num_devices(1);
}
