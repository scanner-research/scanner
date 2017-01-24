#include "scanner/kernels/args.pb.h"
#include "scanner/api/evaluator.h"
#include "scanner/api/kernel.h"
#include "scanner/api/user_function.h"
#include "scanner/util/cuda.h"
#include "scanner/util/memory.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

namespace scanner {

using CustomNetConfiguration =
  void(*)(const FrameInfo& frame_info, caffe::Net<float>* net);

caffe::Caffe::Brew device_type_to_caffe_mode(DeviceType type) {
  caffe::Caffe::Brew caffe_type;

  switch (type) {
    case DeviceType::GPU:
      caffe_type = caffe::Caffe::GPU;
      break;
    case DeviceType::CPU:
      caffe_type = caffe::Caffe::CPU;
      break;
    default:
      // TODO(apoms): error message
      exit(EXIT_FAILURE);
      break;
  }

  return caffe_type;
}

class CaffeKernel : public VideoKernel {
public:
  CaffeKernel(const Kernel::Config& config)
    : VideoKernel(config),
      device_(config.devices[0]) {
    args_.ParseFromArray(config.args.data(), config.args.size());
    UserFunctionRegistry* registry = get_user_function_registry();

    const std::string fn_name = args_.net_configuration_fn();
    if (fn_name.size() > 0) {
      LOG_IF(FATAL, !registry->has_user_function(fn_name))
        << "User function " << fn_name << " has not been registered";
      net_config_ = registry->get_user_function<CustomNetConfiguration>(fn_name);
    }

    set_device();
    // Initialize our network
    auto& descriptor = args_.net_descriptor();
    net_.reset(new caffe::Net<float>(descriptor.model_path(), caffe::TEST));
    net_->CopyTrainedLayersFrom(descriptor.model_weights_path());
    // Initialize memory
    const boost::shared_ptr<caffe::Blob<float>> input_blob{
      net_->blob_by_name(descriptor.input_layer_names(0))};
    input_blob->Reshape({args_.batch_size(), input_blob->shape(1), input_blob->shape(2),
          input_blob->shape(3)});

  }

  void new_frame_info() override {
    i32 frame_width = frame_info_.width;
    i32 frame_height = frame_info_.height;

    set_device();

    auto& descriptor = args_.net_descriptor();
    assert(descriptor.input_layer_names().size() > 0);
    const boost::shared_ptr<caffe::Blob<float>> input_blob{
      net_->blob_by_name(descriptor.input_layer_names(0))};
    if (input_blob->shape(0) != args_.batch_size()) {
      input_blob->Reshape({args_.batch_size(), input_blob->shape(1),
            input_blob->shape(2), input_blob->shape(3)});
    }

    if (net_config_) {
      net_config_(frame_info_, net_.get());
    } else {
      i32 width, height;
      if (descriptor.transpose()) {
        width = frame_height;
        height = frame_width;
      } else {
        width = frame_width;
        height = frame_height;
      }
      if (descriptor.preserve_aspect_ratio()) {
        if (descriptor.input_width() != -1) {
          width = descriptor.input_width();
          f32 scale = static_cast<f32>(descriptor.input_width()) / width;
          width = width * scale;
          height = height * scale;
        } else if (descriptor.input_height() != -1) {
          f32 scale = static_cast<f32>(descriptor.input_height()) / height;
          width = width * scale;
          height = height * scale;
        }
      } else if (descriptor.input_width() != -1) {
        width = descriptor.input_width();
        height = descriptor.input_height();
      }

      if (descriptor.pad_mod() != -1) {
        i32 pad = descriptor.pad_mod();
        width += (width % pad) ? pad - (width % pad) : 0;
        height += (height % pad) ? pad - (height % pad) : 0;
      }

      input_blob->Reshape(
        {input_blob->shape(0), input_blob->shape(1), height, width});
    }
  }

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    set_device();

    auto& descriptor = args_.net_descriptor();
    std::vector<boost::shared_ptr<caffe::Blob<float>>> input_blobs;
    for (const std::string& name : descriptor.input_layer_names()) {
      input_blobs.emplace_back(net_->blob_by_name(name));
    }
    assert(input_blobs.size() > 0);

    size_t num_outputs = descriptor.output_layer_names().size();
    i32 input_count = (i32)input_columns[1].rows.size();
    i32 out_col_idx = 0;
    // forward the frame
    output_columns[out_col_idx].rows = input_columns[0].rows;
    out_col_idx++;
    i32 batch_size = args_.batch_size();
    for (i32 frame = 0; frame < input_count; frame += batch_size) {
      i32 batch_count = std::min(input_count - frame, batch_size);
      if (input_blobs[0]->shape(0) != batch_count) {
        input_blobs[0]->Reshape({batch_count, input_blobs[0]->shape(1),
              input_blobs[0]->shape(2),
              input_blobs[0]->shape(3)});
      }

      for (i32 i = 0; i < input_blobs.size(); ++i) {
        f32* net_input_buffer = nullptr;
        if (device_.type == DeviceType::GPU) {
          net_input_buffer = input_blobs[i]->mutable_gpu_data();
        } else {
          net_input_buffer = input_blobs[i]->mutable_cpu_data();
        }

        size_t offset = 0;
        for (i32 j = 0; j < batch_count; ++j) {
          memcpy_buffer((u8*)net_input_buffer + offset, device_,
                        input_columns[i + 1].rows[frame + j].buffer, device_,
                        input_columns[i + 1].rows[frame + j].size);
          offset += input_columns[i + 1].rows[frame + j].size;
        }
      }

      // Compute features
      auto net_start = now();
      net_->ForwardPrefilled();
      if (profiler_) {
        cudaDeviceSynchronize();
        profiler_->add_interval("caffe:net", net_start, now());
      }

      // Save batch of frames
      size_t total_size = 0;
      i32 total_rows = num_outputs * batch_count;
      for (size_t i = 0; i < num_outputs; ++i) {
        const std::string& output_layer_name = descriptor.output_layer_names(i);
        const boost::shared_ptr<caffe::Blob<float>> output_blob{
          net_->blob_by_name(output_layer_name)};
        size_t output_length = output_blob->count() / batch_count;
        size_t output_size = output_length * sizeof(float);
        total_size += output_size * batch_count;
      }

      u8* output_block = new_block_buffer(device_, total_size, total_rows);
      std::vector<u8*> dest_buffers, src_buffers;
      std::vector<size_t> sizes;
      for (size_t i = 0; i < num_outputs; ++i) {
        const std::string& output_layer_name = descriptor.output_layer_names(i);
        const boost::shared_ptr<caffe::Blob<float>> output_blob{
          net_->blob_by_name(output_layer_name)};
        size_t output_length = output_blob->count() / batch_count;
        size_t output_size = output_length * sizeof(float);
        dest_buffers.push_back(output_block);
        src_buffers.push_back(
          (u8*) (device_.type == DeviceType::CPU
                 ? output_blob->cpu_data()
                 : output_blob->gpu_data()));
        sizes.push_back(output_size * batch_count);
        for (i32 b = 0; b < batch_count; b++) {
          output_columns[out_col_idx + i].rows.push_back(
            Row{output_block, output_size});
          output_block += output_size;
        }
      }

      memcpy_vec(dest_buffers, device_, src_buffers, device_, sizes);
    }
    out_col_idx += num_outputs;
    for (size_t col_idx = input_blobs.size() + 1; col_idx < input_columns.size();
         ++col_idx) {
      output_columns[out_col_idx].rows = input_columns[col_idx].rows;
      out_col_idx++;
    }
  }

  void set_device() {
    caffe::Caffe::set_mode(device_type_to_caffe_mode(device_.type));
    if (device_.type == DeviceType::GPU) {
#ifdef HAVE_CUDA
      // HACK(apoms): caffe does not keep track of device it was initialized
      //  with. For example, if you call cudaSetDevice here before
      //  Caffe::SetDevice, caffe will think the GPU did not change and not
      //  reinit cublas. Need to patch caffe.
      caffe::Caffe::SetDevice(device_.id);
#else
      LOG(FATAL) << "Not built with CUDA support.";
#endif
    }

  }

private:
  DeviceHandle device_;
  proto::CaffeArgs args_;
  CustomNetConfiguration net_config_;
  std::unique_ptr<caffe::Net<float>> net_;
};

}
