#pragma once

#include "scanner/eval/evaluator.h"
#include "scanner/eval/evaluator_constructor.h"

namespace scanner {

class HistogramEvaluator : public Evaluator {
public:
  HistogramEvaluator(EvaluatorConfig config);

  virtual ~HistogramEvaluator();

  virtual void configure(const DatasetItemMetadata& metadata) override;

  virtual void evaluate(
    u8* input_buffer,
    std::vector<std::vector<u8*>>& output_buffers,
    std::vector<std::vector<size_t>>& output_sizes,
    i32 batch_size) override;

private:
  DatasetItemMetadata metadata;
};

class HistogramEvaluatorConstructor : public EvaluatorConstructor {
public:
  HistogramEvaluatorConstructor();

  virtual ~HistogramEvaluatorConstructor();

  virtual i32 get_number_of_devices() override;

  virtual DeviceType get_input_buffer_type() override;

  virtual DeviceType get_output_buffer_type() override;

  virtual i32 get_number_of_outputs() override;

  virtual std::vector<std::string> get_output_names() override;

  virtual u8* new_input_buffer(const EvaluatorConfig& config) override;

  virtual void delete_input_buffer(
    const EvaluatorConfig& config,
    u8* buffer) override;

  virtual void delete_output_buffer(
    const EvaluatorConfig& config,
    u8* buffers) override;

  virtual Evaluator* new_evaluator(const EvaluatorConfig& config) override;

};

}
