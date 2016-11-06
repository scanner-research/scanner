#include "scanner/engine.h"
#include "scanner/evaluators/caffe/caffe_evaluator.h"
#include "scanner/evaluators/caffe/net_descriptor.h"
#include "scanner/evaluators/caffe/squeezenet/squeezenet_input_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

namespace scanner {
namespace {
PipelineDescription get_pipeline_description(
    const DatasetMetadata& dataset_meta,
    const std::vector<DatasetItemMetadata>& item_metas) {
  PipelineDescription desc;
  desc.sampling = Sampling::Strided;
  desc.stride = 10;
  // desc.sampling = Sampling::SequenceGather;
  // desc.gather_sequences.push_back({0, {Interval{183000, 186241}}});

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
}

REGISTER_PIPELINE(knn, get_pipeline_description);
}
