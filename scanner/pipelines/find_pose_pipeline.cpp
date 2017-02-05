#include "scanner/eval/pipeline_description.h"
#include "scanner/evaluators/caffe/caffe_evaluator.h"
#include "scanner/evaluators/caffe/cpm/cpm_input_evaluator.h"
#include "scanner/evaluators/caffe/cpm/cpm_parser_evaluator.h"
#include "scanner/evaluators/caffe/cpm/cpm_person_input_evaluator.h"
#include "scanner/evaluators/caffe/cpm/cpm_person_parser_evaluator.h"
#include "scanner/evaluators/caffe/net_descriptor.h"
#include "scanner/evaluators/util/swizzle_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

namespace scanner {
namespace {
PipelineDescription get_pipeline_description(const DatasetInformation &info) {
  const char *JOB_NAME = std::getenv("SC_JOB_NAME");
  std::string job_name(JOB_NAME);

  PipelineDescription desc;
  Sampler::all(info, desc, job_name, {"centers"});
  Sampler::join_prepend(info, desc, "centers", "frame");

  NetDescriptor cpm_descriptor;
  {
    std::string net_descriptor_file = "features/cpm.toml";
    std::ifstream net_file{net_descriptor_file};
    cpm_descriptor = descriptor_from_net_file(net_file);
  }

  i32 batch_size = 4;
  DeviceType device_type;
  VideoDecoderType decoder_type;
#ifdef HAVE_CUDA
  device_type = DeviceType::GPU;
  decoder_type = VideoDecoderType::NVIDIA;
#else
  device_type = DeviceType::CPU;
  decoder_type = VideoDecoderType::SOFTWARE;
#endif

  std::vector<std::unique_ptr<EvaluatorFactory>> &factories =
      desc.evaluator_factories;

  factories.emplace_back(
      new DecoderEvaluatorFactory(DeviceType::CPU, VideoDecoderType::SOFTWARE));
  // factories.emplace_back(
  //     new DecoderEvaluatorFactory(device_type, decoder_type, 1));
  factories.emplace_back(
      new CPMInputEvaluatorFactory(device_type, cpm_descriptor, batch_size));
  factories.emplace_back(
      new CaffeEvaluatorFactory(device_type, cpm_descriptor, batch_size));
  factories.emplace_back(new CPMParserEvaluatorFactory(device_type));
  factories.emplace_back(
      new SwizzleEvaluatorFactory(device_type, {1}, {"joint_centers"}));

  return desc;
}
}

REGISTER_PIPELINE(find_pose, get_pipeline_description);
}
