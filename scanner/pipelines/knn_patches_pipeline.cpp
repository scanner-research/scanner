#include "scanner/eval/pipeline_description.h"
#include "scanner/evaluators/caffe/caffe_evaluator.h"
#include "scanner/evaluators/caffe/default/default_input_evaluator.h"
#include "scanner/evaluators/caffe/faster_rcnn/faster_rcnn_parser_evaluator.h"
#include "scanner/evaluators/caffe/net_descriptor.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

namespace scanner {
namespace {
PipelineDescription get_pipeline_description(const DatasetInformation &info) {
  PipelineDescription desc;
  Sampler::strided_frames(info, desc, 8);

  std::string net_descriptor_file = "features/faster_rcnn_coco.toml";
  NetDescriptor descriptor;
  {
    std::ifstream net_file{net_descriptor_file};
    descriptor = descriptor_from_net_file(net_file);
  }
  i32 batch_size = 1;

  std::vector<std::unique_ptr<EvaluatorFactory>> &factories =
      desc.evaluator_factories;

  DeviceType device_type;
  VideoDecoderType decoder_type;

#ifdef HAVE_CUDA
  device_type = DeviceType::GPU;
  decoder_type = VideoDecoderType::NVIDIA;
#else
  device_type = DeviceType::CPU;
  decoder_type = VideoDecoderType::SOFTWARE;
#endif

  auto im_info_builder = [=](u8 *&buffer, size_t &size,
                             const InputFormat &metadata) {

    size = 3 * sizeof(f32);
    buffer = new u8[size];
    f32 *blob = (f32 *)buffer;
    *(blob + 0) = metadata.height();
    *(blob + 1) = metadata.width();
    *(blob + 2) = 1.0;
    //     if (device_type == DeviceType::GPU) {
    // #ifdef HAVE_CUDA
    //       u8* gpu_buffer;
    //       cudaMalloc(&gpu_buffer, size);
    //       cudaMemcpy(gpu_buffer, buffer, size, cudaMemcpyHostToDevice);
    //       delete buffer;
    //       buffer = gpu_buffer;
    // #else
    //       LOG(FATAL) << "Cuda not built.";
    // #endif
    //     }
  };

  factories.emplace_back(
      new DecoderEvaluatorFactory(device_type, decoder_type));
  factories.emplace_back(new DefaultInputEvaluatorFactory(
      DeviceType::CPU, descriptor, batch_size, {im_info_builder}));
  factories.emplace_back(
      new CaffeEvaluatorFactory(device_type, descriptor, batch_size));
  factories.emplace_back(new FasterRCNNParserEvaluatorFactory);

  return desc;
}
}

REGISTER_PIPELINE(knn_patches, get_pipeline_description);
}
