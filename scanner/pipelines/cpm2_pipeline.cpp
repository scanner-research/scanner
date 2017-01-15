#include "scanner/engine/runtime.h"
#include "scanner/evaluators/caffe/caffe_evaluator.h"
#include "scanner/evaluators/caffe/cpm2/cpm2_input_evaluator.h"
#include "scanner/evaluators/caffe/cpm2/cpm2_parser_evaluator.h"
#include "scanner/evaluators/caffe/net_descriptor.h"
#include "scanner/evaluators/util/swizzle_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

#include <map>

namespace scanner {
namespace {

PipelineDescription get_pipeline_description(const DatasetInformation& info) {
  char* SCALE = std::getenv("SC_SCALE");
  char* START_FRAME = std::getenv("SC_START_FRAME");
  char* END_FRAME = std::getenv("SC_END_FRAME");

  i32 start_frame = 1000;
  i32 end_frame = 3000;
  if (START_FRAME) {
    start_frame = std::atoi(START_FRAME);
  }
  if (END_FRAME) {
    end_frame = std::atoi(END_FRAME);
  }

  PipelineDescription desc;
  //Sampler::all_frames(info, desc);
  // Sampler::strided_frames(info, desc, 24);
  Sampler::range_frames(info, desc, start_frame, end_frame);

  NetDescriptor cpm_person_descriptor;
  {
    std::string net_descriptor_file = "features/cpm2.toml";
    std::ifstream net_file{net_descriptor_file};
    cpm_person_descriptor = descriptor_from_net_file(net_file);
  }

  f32 scale = 0.25;
  if (SCALE) {
    scale = std::atof(SCALE);
  }

  // CPM2 uses batch size for multiple scales
  i32 batch_size = 1;
  DeviceType device_type;
  VideoDecoderType decoder_type;
#ifdef HAVE_CUDA
  device_type = DeviceType::GPU;
  decoder_type = VideoDecoderType::NVIDIA;
#else
  device_type = DeviceType::CPU;
  decoder_type = VideoDecoderType::SOFTWARE;
#endif

  std::vector<std::unique_ptr<EvaluatorFactory>>& factories =
      desc.evaluator_factories;

  using namespace std::placeholders;
  CustomNetConfiguration net_config = std::bind(cpm2_net_config, scale, _1, _2);

  device_type = DeviceType::CPU;
  decoder_type = VideoDecoderType::SOFTWARE;

  factories.emplace_back(
      new DecoderEvaluatorFactory(device_type, decoder_type));

  device_type = DeviceType::GPU;
  decoder_type = VideoDecoderType::NVIDIA;

  factories.emplace_back(new CPM2InputEvaluatorFactory(
      device_type, cpm_person_descriptor, batch_size, scale));
  factories.emplace_back(new CaffeEvaluatorFactory(
      device_type, cpm_person_descriptor, batch_size, net_config));
  factories.emplace_back(
      new CPM2ParserEvaluatorFactory(DeviceType::CPU, scale));
  factories.emplace_back(
      new SwizzleEvaluatorFactory(DeviceType::CPU, {1}, {"joint_centers"}));

  return desc;
}
}

REGISTER_PIPELINE(cpm2, get_pipeline_description);
}
