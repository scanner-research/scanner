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
    char* input_buffer,
    std::vector<char*> output_buffers,
    int batch_size) override;

private:
  DatasetItemMetadata metadata;
};

class HistogramEvaluatorConstructor : public EvaluatorConstructor {
public:
  HistogramEvaluatorConstructor();

  virtual ~HistogramEvaluatorConstructor();

  virtual int get_number_of_devices() override;

  virtual DeviceType get_input_buffer_type() override;

  virtual DeviceType get_output_buffer_type() override;

  virtual int get_number_of_outputs() override;

  virtual std::vector<std::string> get_output_names() override;

  virtual std::vector<size_t> get_output_element_sizes(
    const EvaluatorConfig& config) override;

  virtual char* new_input_buffer(const EvaluatorConfig& config) override;

  virtual void delete_input_buffer(
    const EvaluatorConfig& config,
    char* buffer) override;

  virtual std::vector<char*> new_output_buffers(
    const EvaluatorConfig& config,
    int num_inputs) override;

  virtual void delete_output_buffers(
    const EvaluatorConfig& config,
    std::vector<char*> buffers) override;

  virtual Evaluator* new_evaluator(const EvaluatorConfig& config) override;

};

}
