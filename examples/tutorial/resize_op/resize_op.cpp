#include "resize.pb.h"            // for ResizeArgs (generated file)
#include "scanner/api/kernel.h"   // for VideoKernel and REGISTER_KERNEL
#include "scanner/api/op.h"       // for REGISTER_OP
#include "scanner/util/memory.h"  // for device-independent memory management
#include "scanner/util/opencv.h"  // for using OpenCV

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
  // from one input column from the video, "frame", and return
  // a single column, "frame".
  void execute(const scanner::BatchedColumns& input_columns,
               scanner::BatchedColumns& output_columns) override {
    auto& frame_column = input_columns[0];
    int input_count = num_rows(frame_column);

    // This must be called at the top of the execute method in any VideoKernel.
    // See the VideoKernel for the implementation check_frame_info.
    check_frame(scanner::CPU_DEVICE, frame_column[0]);

    auto& resized_frame_column = output_columns[0];
    FrameInfo output_frame_info(height_, width_, 3, FrameType::U8);
    for (int i = 0; i < input_count; ++i) {
      // Get a frame from the batch of input frames
      const Frame* frame = frame_column[i].as_const_frame();
      // Convert the input frame into an OpenCV matrix
      cv::Mat input = frame_to_mat(frame);

      // Allocate a frame for the resized output frame
      Frame* resized_frame = new_frame(scanner::CPU_DEVICE, output_frame_info);
      // Convert the output frame to an OpenCV matrix
      cv::Mat output = frame_to_mat(resized_frame);

      // Call to OpenCV for the resize
      cv::resize(input, output, cv::Size(width_, height_));

      // Add the resized frame to an output column
      insert_frame(resized_frame_column, resized_frame);
    }
  }

 private:
  int width_;
  int height_;
};

// These functions run statically when the shared library is loaded to tell the
// Scanner runtime about your custom op.

REGISTER_OP(Resize).frame_input("frame").frame_output("frame");

REGISTER_KERNEL(Resize, ResizeKernel)
    .device(scanner::DeviceType::CPU)
    .num_devices(1);
