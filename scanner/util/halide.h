#pragma once

#include "scanner/util/common.h"
#include "scanner/api/kernel.h"

#include "HalideRuntime.h"

#ifdef HAVE_CUDA
#include "HalideRuntimeCuda.h"
#include "scanner/engine/halide_context.h"
#endif

namespace scanner {

void setup_halide_frame_buf(buffer_t& halide_buf, FrameInfo& frame_info) {
  // Halide has the input format x * stride[0] + y * stride[1] + c * stride[2]
  halide_buf.stride[0] = 3;
  halide_buf.stride[1] = frame_info.width() * 3;
  halide_buf.stride[2] = 1;
  halide_buf.extent[0] = frame_info.width();
  halide_buf.extent[1] = frame_info.height();
  halide_buf.extent[2] = 3;
  halide_buf.elem_size = 1;
}

void set_halide_buf_ptr(const DeviceHandle& device,
                        buffer_t &halide_buf,
                        u8 *buf,
                        size_t size) {
  if (device.type == DeviceType::GPU) {
    CUDA_PROTECT({
      halide_buf.dev = (uintptr_t) nullptr;

      // "You likely want to set the dev_dirty flag for correctness. (It will
      // not matter if all the code runs on the GPU.)"
      halide_buf.dev_dirty = true;

      i32 err =
          halide_cuda_wrap_device_ptr(nullptr, &halide_buf, (uintptr_t)buf);
      LOG_IF(FATAL, err != 0) << "Halide wrap device ptr failed";

      // "You'll need to set the host field of the buffer_t structs to
      // something other than nullptr as that is used to indicate bounds query
      // calls" - Zalman Stern
      halide_buf.host = (u8 *)0xdeadbeef;
      });
  } else {
    halide_buf.host = buf;
  }
}

void unset_halide_buf_ptr(const DeviceHandle& device,
                          buffer_t &halide_buf) {
  if (device.type == DeviceType::GPU) {
    CUDA_PROTECT({ halide_cuda_detach_device_ptr(nullptr, &halide_buf); });
  }
}

}
