#pragma once

#include "scanner/eval/evaluator.h"
#include "scanner/eval/evaluator_factory.h"
#include "scanner/util/opencv.h"

#ifdef HAVE_CUDA
#include "scanner/util/cuda.h"
#endif

namespace scanner {

class HistogramEvaluator : public Evaluator {
 public:
  HistogramEvaluator(DeviceType device_type, i32 device_id);
  ~HistogramEvaluator();

  void configure(const InputFormat& metadata) override;

  void evaluate(const BatchedColumns& input_columns,
                BatchedColumns& output_columns) override;

  void set_device();

 private:
  DeviceType device_type_;
  i32 device_id_;

#ifdef HAVE_CUDA
  cvc::GpuMat hist_;
  std::vector<cvc::GpuMat> planes_;
  cvc::GpuMat out_mat_;
#endif
};

class HistogramEvaluatorFactory : public EvaluatorFactory {
 public:
  HistogramEvaluatorFactory(DeviceType device_type);

  EvaluatorCapabilities get_capabilities() override;

  std::vector<std::string> get_output_names() override;

  Evaluator* new_evaluator(const EvaluatorConfig& config) override;

 private:
  DeviceType device_type_;
};
}
