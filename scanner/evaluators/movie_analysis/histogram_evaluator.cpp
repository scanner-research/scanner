#include "histogram_evaluator.h"
#include "scanner/util/memory.h"

namespace scanner {

const i32 BINS = 16;

HistogramEvaluator::HistogramEvaluator(DeviceType device_type, i32 device_id)
    : device_type_(device_type), device_id_(device_id) {}

HistogramEvaluator::~HistogramEvaluator() {}

void HistogramEvaluator::configure(const BatchConfig& config) {
  config_ = config;
  set_device();

  if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
    hist_ = cvc::GpuMat(1, BINS, CV_32S);
    for (i32 i = 0; i < 3; ++i) {
      planes_.push_back(
          cvc::GpuMat(config_.formats[0].height(), config_.formats[0].width(), CV_8UC1));
    }
    out_mat_ = cvc::GpuMat(1, BINS * 3, CV_32S);
#endif
  }
}

void HistogramEvaluator::evaluate(const BatchedColumns& input_columns,
                                  BatchedColumns& output_columns) {
  assert(input_columns.size() == 1);
  set_device();

  i64 hist_size = BINS * 3 * sizeof(float);
  i32 input_count = (i32)input_columns[0].rows.size();
  u8* output_block = new_block_buffer({device_type_, device_id_},
                                      hist_size * input_count,
                                      input_count);

  if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
    for (i32 i = 0; i < input_count; ++i) {
      cvc::GpuMat img =
          bytesToImage_gpu(input_columns[0].rows[i].buffer, config_.formats[0]);
      cvc::split(img, planes_);

      for (i32 j = 0; j < 3; ++j) {
        cvc::histEven(planes_[j], hist_, BINS, 0, 256);
        hist_.copyTo(out_mat_(cv::Rect(j * BINS, 0, BINS, 1)));
      }

      u8* output_buf = output_block + i * hist_size;
      cudaMemcpy(output_buf, out_mat_.data, hist_size,
                 cudaMemcpyDeviceToDevice);
      output_columns[0].rows.push_back(Row{output_buf, hist_size});
    }
#else
    LOG(FATAL) << "Cuda not installed.";
#endif
  } else {
    for (i32 i = 0; i < input_count; ++i) {
      cv::Mat img = bytesToImage(input_columns[0].rows[i].buffer,
                                 config_.formats[0]);

      std::vector<cv::Mat> bgr_planes;
      cv::split(img, bgr_planes);

      cv::Mat r_hist, g_hist, b_hist;
      float range[] = {0, 256};
      const float* histRange = {range};

      cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &BINS,
                   &histRange);
      cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &BINS,
                   &histRange);
      cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &BINS,
                   &histRange);

      std::vector<cv::Mat> hists = {r_hist, g_hist, b_hist};
      cv::Mat hist;
      cv::hconcat(hists, hist);

      u8* hist_buffer = output_block + i * hist_size;
      assert(hist_size == hist.total() * hist.elemSize());
      memcpy(hist_buffer, hist.data, hist_size);

      output_columns[0].rows.push_back(Row{hist_buffer, hist_size});
    }
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
  const std::vector<std::string>& input_columns)
{
  return {"histogram"};
}

Evaluator* HistogramEvaluatorFactory::new_evaluator(
    const EvaluatorConfig& config) {
  return new HistogramEvaluator(device_type_, config.device_ids[0]);
}
}
