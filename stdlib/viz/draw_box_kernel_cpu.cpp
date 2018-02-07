#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"
#include "scanner/util/serialize.h"

namespace scanner {

class DrawBoxKernelCPU : public BatchedKernel {
 public:
  DrawBoxKernelCPU(const KernelConfig& config)
    : BatchedKernel(config), device_(config.devices[0]) {}

  void execute(const BatchedElements& input_columns,
               BatchedElements& output_columns) override {
    auto& frame_col = input_columns[0];
    auto& bbox_col = input_columns[1];

    i32 input_count = num_rows(frame_col);
    FrameInfo info = frame_col[0].as_const_frame()->as_frame_info();
    std::vector<Frame*> output_frames = new_frames(device_, info, input_count);

    for (i32 i = 0; i < input_count; ++i) {
      cv::Mat img = frame_to_mat(frame_col[i].as_const_frame());
      cv::Mat out_img = frame_to_mat(output_frames[i]);
      img.copyTo(out_img);

      // Deserialize bboxes
      std::vector<BoundingBox> bboxes =
          deserialize_bbox_vector(bbox_col[i].buffer, bbox_col[i].size);

      // Draw all bboxes
      for (auto& bbox : bboxes) {
        i32 width = bbox.x2() - bbox.x1();
        i32 height = bbox.y2() - bbox.y1();
        cv::rectangle(out_img, cv::Rect(bbox.x1(), bbox.y1(), width, height),
                      cv::Scalar(255, 0, 0), 2);
      }
      insert_frame(output_columns[0], output_frames[i]);
    }
  }

 private:
  DeviceHandle device_;
};

REGISTER_OP(DrawBox).frame_input("frame").input("bboxes").frame_output("frame");

REGISTER_KERNEL(DrawBox, DrawBoxKernelCPU)
    .device(DeviceType::CPU)
    .num_devices(1);
}
