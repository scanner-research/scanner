#include "scanner/eval/pipeline_description.h"
#include "scanner/evaluators/caffe/caffe_evaluator.h"
#include "scanner/evaluators/caffe/default/default_input_evaluator.h"
#include "scanner/evaluators/caffe/net_descriptor.h"
#include "scanner/evaluators/util/discard_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"
#include "scanner/evaluators/util/swizzle_evaluator.h"

namespace scanner {
namespace {
PipelineDescription get_pipeline_description(const DatasetInformation& info) {
  PipelineDescription desc;
  Sampler::all_frames(info, desc);
  // desc.sampling = Sampling::Strided;

  // std::ifstream infile("stride.txt");
  // i32 stride;
  // infile >> stride;
  // desc.stride = stride;

  std::string net_descriptor_file = "features/googlenet.toml";
  NetDescriptor descriptor;
  {
    std::ifstream net_file{net_descriptor_file};
    descriptor = descriptor_from_net_file(net_file);
  }
  i32 batch_size = 100;

  DeviceType device_type;
  VideoDecoderType decoder_type;

#if HAVE_CUDA
  device_type = DeviceType::GPU;
  decoder_type = VideoDecoderType::NVIDIA;
#else
  device_type = DeviceType::CPU;
  decoder_type = VideoDecoderType::SOFTWARE;
#endif

  std::vector<std::unique_ptr<EvaluatorFactory>>& factories =
      desc.evaluator_factories;

  factories.emplace_back(
      new DecoderEvaluatorFactory(device_type, decoder_type));
  factories.emplace_back(new DefaultInputEvaluatorFactory(
      device_type, descriptor, batch_size));
  factories.emplace_back(
      new CaffeEvaluatorFactory(device_type, descriptor, batch_size));
  factories.emplace_back(new DiscardEvaluatorFactory(device_type));

  return desc;
}
}

REGISTER_PIPELINE(knn, get_pipeline_description);
}
