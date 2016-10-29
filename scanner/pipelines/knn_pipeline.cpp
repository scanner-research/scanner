#include "scanner/engine.h"
#include "scanner/evaluators/caffe/caffe_evaluator.h"
#include "scanner/evaluators/caffe/net_descriptor.h"
#include "scanner/evaluators/caffe/squeezenet/squeezenet_input_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

using namespace scanner;

PipelineDescription get_pipeline_description() {
  PipelineDescription desc;
  desc.sampling = PipelineDescription::Sampling::Strided;
  desc.stride = 4;

  std::string net_descriptor_file = "features/squeezenet.toml";
  NetDescriptor descriptor;
  {
    std::ifstream net_file{net_descriptor_file};
    descriptor = descriptor_from_net_file(net_file);
  }
  i32 batch_size = 96;

  std::vector<std::unique_ptr<EvaluatorFactory>>& factories =
      desc.evaluator_factories;

  factories.emplace_back(
      new DecoderEvaluatorFactory(DeviceType::CPU, VideoDecoderType::SOFTWARE));
  factories.emplace_back(new SqueezeNetInputEvaluatorFactory(
      DeviceType::CPU, descriptor, batch_size));
  factories.emplace_back(new CaffeEvaluatorFactory(DeviceType::GPU, descriptor,
                                                   batch_size, false));

  return desc;
}
