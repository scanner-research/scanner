#include "scanner/eval/pipeline_description.h"
#include "scanner/evaluators/caffe/caffe_evaluator.h"
#include "scanner/evaluators/caffe/default/default_input_evaluator.h"
#include "scanner/evaluators/util/discard_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

#include <cstdlib>

namespace scanner {
namespace {
PipelineDescription get_pipeline_description(const DatasetInformation& info) {
  const char* NET = std::getenv("SC_NET");
  const char* BATCH_SIZE = std::getenv("SC_BATCH_SIZE");

  std::string net_descriptor_file = std::string(NET);
  i32 batch_size = std::atoi(BATCH_SIZE);

  NetDescriptor descriptor;
  {
    std::ifstream net_file{net_descriptor_file};
    descriptor = descriptor_from_net_file(net_file);
  }

  PipelineDescription desc;
  Sampler::all_frames(info, desc);

  std::vector<std::unique_ptr<EvaluatorFactory>>& factories =
      desc.evaluator_factories;

  factories.emplace_back(
      new DecoderEvaluatorFactory(DeviceType::CPU, VideoDecoderType::SOFTWARE));
  factories.emplace_back(new DefaultInputEvaluatorFactory(
      DeviceType::GPU, descriptor, batch_size));
  factories.emplace_back(
      new CaffeEvaluatorFactory(DeviceType::GPU, descriptor, batch_size));
  factories.emplace_back(new DiscardEvaluatorFactory(DeviceType::GPU));

  return desc;
}
}

REGISTER_PIPELINE(dnn_rate, get_pipeline_description);
}
