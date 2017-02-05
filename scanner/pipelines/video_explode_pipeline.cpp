#include "scanner/eval/pipeline_description.h"
#include "scanner/evaluators/image/image_encoder_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

namespace scanner {
namespace {
PipelineDescription get_pipeline_description(const DatasetInformation &info) {
  const char *ENCODING = std::getenv("SC_ENCODING");

  PipelineDescription desc;
  Sampler::all_frames(info, desc);

  ImageEncodingType image_type = ImageEncodingType::JPEG;
  if (ENCODING) {
    bool result =
        string_to_image_encoding_type(std::string(ENCODING), image_type);
    assert(result);
  }

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
      new DecoderEvaluatorFactory(device_type, decoder_type));
  factories.emplace_back(
      new ImageEncoderEvaluatorFactory(DeviceType::CPU, image_type));

  return desc;
}
}

REGISTER_PIPELINE(kaboom, get_pipeline_description);
}
