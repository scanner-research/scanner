#include "scanner/api/op.h"      // for REGISTER_OP
#include "scanner/api/kernel.h"  // for VideoKernel and REGISTER_KERNEL
#include "scanner/util/opencv.h" // for using OpenCV
#include "scanner/util/memory.h" // for device-independent memory management
#include "resize.pb.h"           // for ResizeArgs (generated file)

/*
 * Ops in Scanner are abstract units of computation that are implemented by
 * kernels. Kernels are pinned to a specific device (CPU or GPU). Here, we
 * implement a custom op to resize an image. After reading this file, look
 * at CMakeLists.txt for how to build the op.
 */

// Custom kernels must inherit the Kernel class or any subclass thereof,
// e.g. the VideoKernel which provides support for processing video frames.
class ResizeKernel : public scanner::VideoKernel {
public:

  // To allow ops to be customized by users at a runtime, e.g. to define the
  // target width and height of the ResizeKernel, Scanner uses Google's Protocol
  // Buffers, or protobufs, to define serialzable types usable in C++ and
  // Python (see resize_op/args.proto). By convention, ops that take
  // arguments must define a protobuf called <OpName>Args, e.g. ResizeArgs,
  // In Python, users will provide the argument fields to the op constructor,
  // and these will get serialized into a string. This string is part of the
  // general configuration each kernel receives from the runtime, config.args.
  ResizeKernel(const scanner::Kernel::Config& config)
    : scanner::VideoKernel(config) {
    // The protobuf arguments must be decoded from the input string.
    ResizeArgs args;
    args.ParseFromArray(config.args.data(), config.args.size());
    width_ = args.width();
    height_ = args.height();
  }

  // Execute is the core computation of the kernel. It maps a batch of rows
  // from an input table to a batch of rows of the output table. Here, we map
  // from two input columns from the video, "frame" and "frame_info", and return
  // a single column, "frame".
  void execute(const scanner::BatchedColumns &input_columns,
               scanner::BatchedColumns &output_columns) override {
    int input_count = input_columns[0].rows.size();

    // This must be called at the top of the execute method in any VideoKernel.
    // See the VideoKernel for the implementation check_frame_info.
    check_frame_info(scanner::CPU_DEVICE, input_columns[1]);

    for (int i = 0; i < input_count; ++i) {
      // Convert the raw input buffer into an OpenCV matrix
      cv::Mat input(
        frame_info_.height(),
        frame_info_.width(),
        CV_8UC3,
        input_columns[0].rows[i].buffer);

      // Allocate a buffer for the output
      size_t output_size = width_ * height_ * 3;
      unsigned char* output_buf =
        scanner::new_buffer(scanner::CPU_DEVICE, output_size);
      cv::Mat output(height_, width_, CV_8UC3, output_buf);

      // Call to OpenCV for the resize
      cv::resize(input, output, cv::Size(width_, height_));

      // Add the buffer to an output column
      INSERT_ROW(output_columns[0], output_buf, output_size);
    }
  }

private:
  int width_;
  int height_;
};

// These functions run statically when the shared library is loaded to tell the
// Scanner runtime about your custom op.

REGISTER_OP(Resize).inputs({"frame", "frame_info"}).outputs({"frame"});

REGISTER_KERNEL(Resize, ResizeKernel)
   .device(scanner::DeviceType::CPU)
   .num_devices(1);
