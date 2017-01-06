#pragma once

#include "scanner/eval/evaluator.h"
#include "scanner/eval/evaluator_factory.h"
#include "scanner/util/opencv.h"

#ifdef HAVE_CUDA
#include "scanner/util/cuda.h"
#endif

namespace scanner {

class OpticalFlowEvaluator : public Evaluator {
 public:
  OpticalFlowEvaluator(DeviceType device_type, i32 device_id);
  ~OpticalFlowEvaluator();

  void reset() override;

  void evaluate(const BatchedColumns& input_columns,
                BatchedColumns& output_columns) override;

  void set_device();

 private:
  DeviceType device_type_;
  i32 device_id_;
  void* initial_frame_;

#ifdef HAVE_CUDA
  i32 num_cuda_streams_;
  std::vector<cv::cuda::Stream> streams_;
  cvc::GpuMat flow;
#endif
};

class OpticalFlowEvaluatorFactory : public EvaluatorFactory {
 public:
  OpticalFlowEvaluatorFactory(DeviceType device_type);

  EvaluatorCapabilities get_capabilities() override;

  std::vector<std::string> get_output_columns(
    const std::vector<std::string>& input_columns) override;

  Evaluator* new_evaluator(const EvaluatorConfig& config) override;

 private:
  DeviceType device_type_;
};
}
