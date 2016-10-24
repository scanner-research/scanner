#include "scanner/evaluators/caffe/caffe_evaluator.h"
#include "scanner/evaluators/caffe/net_descriptor.h"
#include "scanner/evaluators/caffe/yolo/yolo_input_evaluator.h"

using namespace scanner;

std::vector<std::unique_ptr<EvaluatorFactory>> setup_evaluator_pipeline() {
  std::string net_descriptor_file = "features/yolo.toml";
  NetDescriptor descriptor;
  {
    std::ifstream net_file{net_descriptor_file};
    descriptor = descriptor_from_net_file(net_file);
  }
  i32 batch_size = 24;

  std::vector<std::unique_ptr<EvaluatorFactory>> factories;

  factories.emplace_back(
      new YoloInputEvaluatorFactory(DeviceType::CPU, descriptor, batch_size));
  factories.emplace_back(new CaffeEvaluatorFactory(DeviceType::GPU, descriptor,
                                                   batch_size, false));

  return factories;
}
