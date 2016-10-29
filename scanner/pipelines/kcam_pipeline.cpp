#include "scanner/engine.h"
#include "scanner/evaluators/video/decoder_evaluator.h"
#include "scanner/evaluators/caffe/caffe_evaluator.h"
#include "scanner/evaluators/caffe/facenet/facenet_input_evaluator.h"
#include "scanner/evaluators/caffe/facenet/facenet_parser_evaluator.h"
#include "scanner/evaluators/caffe/yolo/yolo_input_evaluator.h"
#include "scanner/evaluators/caffe/net_descriptor.h"
#include "scanner/evaluators/tracker/tracker_evaluator.h"
#include "scanner/evaluators/util/swizzle_evaluator.h"

using namespace scanner;

PipelineDescription get_pipeline_description() {
  PipelineDescription desc;
  // desc.sampling = Sampling::Gather;
  // PointSamples samples;
  // samples.video_index = 0;
  // for (i32 i = 100; i < 300; ++i) {
  //   samples.frames.push_back(i);
  // }
  // desc.gather_points = {samples};

  std::string net_descriptor_file = "features/caffe_facenet.toml";
  NetDescriptor descriptor;
  {
    std::ifstream net_file{net_descriptor_file};
    descriptor = descriptor_from_net_file(net_file);
  }
  i32 batch_size = 4;

  std::vector<std::unique_ptr<EvaluatorFactory>>& factories =
      desc.evaluator_factories;

  factories.emplace_back(
      new DecoderEvaluatorFactory(DeviceType::CPU, VideoDecoderType::SOFTWARE));
  factories.emplace_back(new FacenetInputEvaluatorFactory(
      DeviceType::GPU, descriptor, batch_size));
  factories.emplace_back(new CaffeEvaluatorFactory(
      DeviceType::GPU, descriptor, batch_size, true));
  factories.emplace_back(new FacenetParserEvaluatorFactory(
      DeviceType::CPU, 0.5, FacenetParserEvaluator::NMSType::Average, true));
  factories.emplace_back(new TrackerEvaluatorFactory(DeviceType::CPU, 32));
  factories.emplace_back(new SwizzleEvaluatorFactory(
      DeviceType::CPU, {1, 2}, {"base_bboxes", "tracked_bboxes"}));

  return desc;
}
