#include "histogram_evaluator.h"
#include "scanner/util/memory.h"

namespace scanner {

const i32 BINS = 16;

HistogramEvaluator::HistogramEvaluator(DeviceType device_type, i32 device_id)
    : device_type_(device_type), device_id_(device_id)
#ifdef HAVE_CUDA
      ,
      num_cuda_streams_(32), streams_(num_cuda_streams_)
#endif
{
}

HistogramEvaluator::~HistogramEvaluator() {}

void HistogramEvaluator::configure(const BatchConfig &config) {
  config_ = config;
  set_device();

  format_ = config_.formats[0];

  if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
    streams_.resize(0);
    streams_.resize(num_cuda_streams_);
    planes_.clear();
    for (i32 i = 0; i < 3; ++i) {
      planes_.push_back(
          cvc::GpuMat(format_.height(), format_.width(), CV_8UC1));
    }
#endif
  }
}

void HistogramEvaluator::evaluate(const BatchedColumns &input_columns,
                                  BatchedColumns &output_columns) {
  assert(input_columns.size() == 1);
  set_device();

  size_t hist_size = BINS * 3 * sizeof(float);
  i32 input_count = (i32)input_columns[0].rows.size();
  u8 *output_block = new_block_buffer({device_type_, device_id_},
                                      hist_size * input_count, input_count);

  auto start = now();

  if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
    for (i32 i = 0; i < input_count; ++i) {
      i32 sid = i % num_cuda_streams_;
      cv::cuda::Stream &s = streams_[sid];

      cvc::GpuMat img =
          bytesToImage_gpu(input_columns[0].rows[i].buffer, format_);
      cvc::split(img, planes_, s);

      u8 *output_buf = output_block + i * hist_size;
      cvc::GpuMat out_mat(1, BINS * 3, CV_32S, output_buf);

      for (i32 j = 0; j < 3; ++j) {
        cvc::histEven(planes_[j], out_mat(cv::Rect(j * BINS, 0, BINS, 1)), BINS,
                      0, 256, s);
      }

      output_columns[0].rows.push_back(Row{output_buf, hist_size});
    }

    for (cv::cuda::Stream &s : streams_) {
      s.waitForCompletion();
    }
#else
    LOG(FATAL) << "Cuda not installed.";
#endif
  } else {
    cv::Mat tmp;
    for (i32 i = 0; i < input_count; ++i) {
      cv::Mat img = bytesToImage(input_columns[0].rows[i].buffer, format_);

      float range[] = {0, 256};
      const float *histRange = {range};

      u8 *output_buf = output_block + i * hist_size;

      for (i32 j = 0; j < 3; ++j) {
        int channels[] = {j};
        cv::Mat out(BINS, 1, CV_32S, output_buf + BINS * sizeof(float));
        cv::calcHist(&img, 1, channels, cv::Mat(), out, 1, &BINS, &histRange);
      }

      output_columns[0].rows.push_back(Row{output_buf, hist_size});
    }
  }

  if (profiler_) {
    profiler_->add_interval("histogram", start, now());
  }
}

void HistogramEvaluator::set_device() {
  if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
    CU_CHECK(cudaSetDevice(device_id_));
    cvc::setDevice(device_id_);
#endif
  }
}

HistogramEvaluatorFactory::HistogramEvaluatorFactory(DeviceType device_type)
    : device_type_(device_type) {}

EvaluatorCapabilities HistogramEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = device_type_;
  caps.max_devices = device_type_ == DeviceType::GPU
                         ? 1
                         : EvaluatorCapabilities::UnlimitedDevices;
  caps.warmup_size = 0;
  return caps;
}

std::vector<std::string> HistogramEvaluatorFactory::get_output_columns(
    const std::vector<std::string> &input_columns) {
  return {"histogram"};
}

Evaluator *
HistogramEvaluatorFactory::new_evaluator(const EvaluatorConfig &config) {
  return new HistogramEvaluator(device_type_, config.device_ids[0]);
}
}
