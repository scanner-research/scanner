#include "scanner/eval/pipeline_description.h"
#include "scanner/evaluators/caffe/caffe_evaluator.h"
#include "scanner/evaluators/caffe/facenet/facenet_input_evaluator.h"
#include "scanner/evaluators/caffe/facenet/facenet_parser_evaluator.h"
#include "scanner/evaluators/caffe/net_descriptor.h"
#include "scanner/evaluators/caffe/yolo/yolo_input_evaluator.h"
#include "scanner/evaluators/tracker/tracker_evaluator.h"
#include "scanner/evaluators/util/swizzle_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

namespace scanner {
namespace {
PipelineDescription get_pipeline_description(
    const DatasetMetadata& dataset_desc,
    const std::vector<DatasetItemMetadata>& item_descriptors) {
  PipelineDescription desc;
  desc.input_columns = {"frame"};
  desc.sampling = Sampling::SequenceGather;
  desc.gather_sequences = {{0, {StridedInterval{18250, 19150}}}};

  std::string net_descriptor_file = "features/caffe_facenet.toml";
  NetDescriptor descriptor;
  {
    std::ifstream net_file{net_descriptor_file};
    descriptor = descriptor_from_net_file(net_file);
  }
  i32 batch_size = 4;

  std::vector<std::unique_ptr<EvaluatorFactory>>& factories =
      desc.evaluator_factories;

  // factories.emplace_back(
  //     new DecoderEvaluatorFactory(DeviceType::GPU, VideoDecoderType::NVIDIA));
  factories.emplace_back(
      new DecoderEvaluatorFactory(DeviceType::CPU, VideoDecoderType::SOFTWARE));
  factories.emplace_back(new FacenetInputEvaluatorFactory(
      DeviceType::GPU, descriptor, batch_size));
  factories.emplace_back(
      new CaffeEvaluatorFactory(DeviceType::GPU, descriptor, batch_size, true));
  factories.emplace_back(new FacenetParserEvaluatorFactory(
      DeviceType::CPU, 0.8, FacenetParserEvaluator::NMSType::Average, true));
  factories.emplace_back(new TrackerEvaluatorFactory(DeviceType::CPU, 32));
  factories.emplace_back(new SwizzleEvaluatorFactory(
      DeviceType::CPU, {1, 2}, {"base_bboxes", "tracked_bboxes"}));

  return desc;
}
}

REGISTER_PIPELINE(kcam, get_pipeline_description);
}
