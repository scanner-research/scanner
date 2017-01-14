#include "scanner/eval/pipeline_description.h"
#include "scanner/evaluators/caffe/caffe_evaluator.h"
#include "scanner/evaluators/caffe/facenet/facenet_input_evaluator.h"
#include "scanner/evaluators/caffe/facenet/facenet_parser_evaluator.h"
#include "scanner/evaluators/caffe/net_descriptor.h"
#include "scanner/evaluators/util/swizzle_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

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
    end_frame = std::atoi(START_FRAME);
  }

  PipelineDescription desc;
  //Sampler::all_frames(info, desc);
  Sampler::range_frames(info, desc, start_frame, end_frame);

  std::string net_descriptor_file = "features/caffe_facenet.toml";
  NetDescriptor descriptor;
  {
    std::ifstream net_file{net_descriptor_file};
    descriptor = descriptor_from_net_file(net_file);
  }
  i32 batch_size = 2;

  f32 scale = 0.25;
  f32 threshold = 0.5;

  if (SCALE) {
    scale = std::atof(SCALE);
  }

  auto facenet_net_config =
      [scale](const BatchConfig &config, caffe::Net<float> *net) {
        assert(config.formats.size() == 1);
        const InputFormat &metadata = config.formats[0];
        // Calculate width by scaling by box size
        int resize_width = metadata.width() * scale;
        int resize_height = metadata.height() * scale;

        resize_width += (resize_width % 8);
        resize_height += (resize_height % 8);

        int net_input_width = resize_height;
        int net_input_height = resize_width;

        const boost::shared_ptr<caffe::Blob<float>> input_blob{
            net->blob_by_name("data")};
        input_blob->Reshape({input_blob->shape(0), input_blob->shape(1),
                             net_input_height, net_input_width});
  };

  std::vector<std::unique_ptr<EvaluatorFactory>> &factories =
      desc.evaluator_factories;

  factories.emplace_back(
      new DecoderEvaluatorFactory(DeviceType::GPU, VideoDecoderType::NVIDIA));
  factories.emplace_back(new FacenetInputEvaluatorFactory(
      DeviceType::GPU, descriptor, batch_size, scale));
  factories.emplace_back(new CaffeEvaluatorFactory(
      DeviceType::GPU, descriptor, batch_size, facenet_net_config));
  factories.emplace_back(
      new FacenetParserEvaluatorFactory(DeviceType::CPU, scale, threshold,
                                        FacenetParserEvaluator::NMSType::Best));
  factories.emplace_back(new SwizzleEvaluatorFactory(
      DeviceType::CPU, {1}, {"base_bboxes",}));

  return desc;
}
}

REGISTER_PIPELINE(facenet, get_pipeline_description);
}
