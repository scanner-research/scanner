#include "stdlib/caffe/caffe_input_kernel.h"
#include "scanner/util/memory.h"

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"

#ifdef HAVE_CUDA
#include "HalideRuntimeCuda.h"
#include "scanner/util/halide_context.h"
#endif

namespace scanner {

CaffeInputKernel::CaffeInputKernel(const Kernel::Config& config)
  : VideoKernel(config), device_(config.devices[0]) {
  args_.ParseFromArray(config.args.data(), config.args.size());
  if (device_.type == DeviceType::GPU) {
    CUDA_PROTECT({
      CUD_CHECK(cuDevicePrimaryCtxRetain(&context_, device_.id));
      Halide::Runtime::Internal::Cuda::context = context_;
      halide_set_gpu_device(device_.id);
    });
  }
}

CaffeInputKernel::~CaffeInputKernel() {
  CUDA_PROTECT({
    cudaSetDevice(device_.id);
    Halide::Runtime::Internal::Cuda::context = 0;
    CUD_CHECK(cuDevicePrimaryCtxRelease(device_.id));
  });
}

void CaffeInputKernel::new_frame_info() {
  if (args_.net_descriptor().input_width() == -1) {
    net_input_width_ = frame_info_.shape[1];
    net_input_height_ = frame_info_.shape[2];
  } else {
    net_input_width_ = args_.net_descriptor().input_width();
    net_input_height_ = args_.net_descriptor().input_height();
  }
}

void CaffeInputKernel::set_halide_buf(buffer_t& halide_buf, u8* buf,
                                      size_t size) {
  if (device_.type == DeviceType::GPU) {
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
      halide_buf.host = (u8*)0xdeadbeef;
    });
  } else {
    halide_buf.host = buf;
  }
}

void CaffeInputKernel::unset_halide_buf(buffer_t& halide_buf) {
  if (device_.type == DeviceType::GPU) {
    CUDA_PROTECT({ halide_cuda_detach_device_ptr(nullptr, &halide_buf); });
  }
}

void CaffeInputKernel::transform_halide(const u8* input_buffer,
                                        u8* output_buffer) {
  i32 frame_width = frame_info_.shape[1];
  i32 frame_height = frame_info_.shape[2];
  size_t net_input_size =
    net_input_width_ * net_input_height_ * 3 * sizeof(float);

  buffer_t input_buf = {0}, output_buf = {0};

  set_halide_buf(input_buf, const_cast<u8*>(input_buffer),
                 frame_width * frame_height * 3);
  set_halide_buf(output_buf, output_buffer, net_input_size);

  // Halide has the input format x * stride[0] + y * stride[1] + c * stride[2]
  // input_buf.host = input_buffer;
  input_buf.stride[0] = 3;
  input_buf.stride[1] = frame_width * 3;
  input_buf.stride[2] = 1;
  input_buf.extent[0] = frame_width;
  input_buf.extent[1] = frame_height;
  input_buf.extent[2] = 3;
  input_buf.elem_size = 1;

  // Halide conveniently defaults to a planar format, which is what Caffe
  // expects
  output_buf.host = output_buffer;
  output_buf.stride[0] = 1;
  output_buf.stride[1] = net_input_width_;
  output_buf.stride[2] = net_input_width_ * net_input_height_;
  output_buf.extent[0] = net_input_width_;
  output_buf.extent[1] = net_input_height_;
  output_buf.extent[2] = 3;
  output_buf.elem_size = 4;

  decltype(caffe_input_transformer_cpu)* func;
  if (device_.type == DeviceType::GPU) {
    CUDA_PROTECT({ func = caffe_input_transformer_gpu; });
  } else {
    func = caffe_input_transformer_cpu;
  }
  auto descriptor = args_.net_descriptor();
  int error =
    func(&input_buf, frame_width, frame_height, net_input_width_,
         net_input_height_, descriptor.normalize(), descriptor.mean_colors(2),
         descriptor.mean_colors(1), descriptor.mean_colors(0), &output_buf);
  LOG_IF(FATAL, error != 0) << "Halide error " << error;

  unset_halide_buf(input_buf);
  unset_halide_buf(output_buf);
}

void CaffeInputKernel::transform_caffe(u8* input_buffer, u8* output_buffer) {
  i32 frame_width = frame_info_.shape[1];
  i32 frame_height = frame_info_.shape[2];
  size_t net_input_size =
    net_input_width_ * net_input_height_ * 3 * sizeof(float);

  cv::Mat input_mat(frame_height, frame_width, CV_8UC3, input_buffer);
  cv::Mat resized_input;

  cv::resize(input_mat, resized_input,
             cv::Size(net_input_width_, net_input_height_), 0, 0,
             cv::INTER_LINEAR);
  cv::cvtColor(resized_input, resized_input, CV_RGB2BGR);
  std::vector<cv::Mat> input_mats = {resized_input};

  caffe::Blob<f32> output_blob;
  output_blob.Reshape(1, 3, net_input_height_, net_input_width_);
  output_blob.set_cpu_data((f32*)output_buffer);

  caffe::TransformationParameter param;
  auto& descriptor = args_.net_descriptor();
  auto& mean_colors = descriptor.mean_colors();
  param.set_force_color(true);
  if (descriptor.normalize()) {
    param.set_scale(1.0 / 255.0);
  }
  for (i32 i = 0; i < mean_colors.size(); i++) {
    param.add_mean_value(mean_colors.Get(i));
  }

  caffe::DataTransformer<f32> transformer(param, caffe::TEST);
  transformer.Transform(input_mats, &output_blob);
}

void CaffeInputKernel::execute(const BatchedColumns& input_columns,
                               BatchedColumns& output_columns) {
  auto& frame_col = input_columns[0];
  check_frame(device_, frame_col[0]);

  auto eval_start = now();
  i32 input_count = NUM_ROWS(frame_col);
  size_t net_input_size =
    net_input_width_ * net_input_height_ * 3 * sizeof(float);

  set_device();

  u8* output_block =
    new_block_buffer(device_, net_input_size * input_count, input_count);

  for (i32 frame = 0; frame < input_count; frame++) {
    const u8* input_buffer = frame_col[frame].as_const_frame()->data;
    u8* output_buffer = output_block + frame * net_input_size;

    transform_halide(input_buffer, output_buffer);

    INSERT_ELEMENT(output_columns[0], output_buffer, net_input_size);
  }

  extra_inputs(input_columns, output_columns);

  if (profiler_) {
    profiler_->add_interval("caffe:transform_input", eval_start, now());
  }
}

void CaffeInputKernel::set_device() {
  CUDA_PROTECT({
    cv::cuda::setDevice(device_.id);
    CU_CHECK(cudaSetDevice(device_.id));
    halide_set_gpu_device(device_.id);
  });
}
}
