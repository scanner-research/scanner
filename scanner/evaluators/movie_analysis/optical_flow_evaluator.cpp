#include "optical_flow_evaluator.h"

#include <cstring>
#include <opencv2/video.hpp>
#include "scanner/util/cycle_timer.h"
#include "scanner/util/memory.h"

namespace scanner {

OpticalFlowEvaluator::OpticalFlowEvaluator(DeviceType device_type, i32 device_id)
  : device_type_(device_type), device_id_(device_id), initial_frame_(nullptr)
#ifdef HAVE_CUDA
  ,
    num_cuda_streams_(4),
    streams_(num_cuda_streams_)
#endif
{
  set_device();
  if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
    streams_.resize(0);
    streams_.resize(num_cuda_streams_);
#else
    // TODO(wcrichto): can we make this #ifdef HAVE_CUDA pattern generic?
    LOG(FATAL) << "Not built with CUDA support.";
#endif
  }
  reset();
}

OpticalFlowEvaluator::~OpticalFlowEvaluator() {
  set_device();
  if (initial_frame_ != nullptr) {
    if (device_type_ == DeviceType::GPU) {
      delete ((cvc::GpuMat*)initial_frame_);
    } else {
      delete ((cv::Mat*)initial_frame_);
    }
  }
}

void OpticalFlowEvaluator::reset() {
  set_device();
  if (device_type_ == DeviceType::GPU) {
    if (initial_frame_ != nullptr) {
      delete ((cvc::GpuMat*)initial_frame_);
    }
    initial_frame_ = (void*)new cvc::GpuMat();
  } else {
    if (initial_frame_ != nullptr) {
      delete ((cv::Mat*)initial_frame_);
    }
    initial_frame_ = (void*)new cv::Mat();
  }
}

void OpticalFlowEvaluator::evaluate(const BatchedColumns& input_columns,
                                    BatchedColumns& output_columns) {
  i32 input_count = (i32)input_columns[0].rows.size();
  size_t out_buf_size =
    config_.formats[0].width() * config_.formats[0].height() * 2 * sizeof(float);

  u8* output_block = new_block_buffer({device_type_, device_id_},
                                      out_buf_size * input_count,
                                      input_count);

  set_device();

  if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
    std::vector<cvc::GpuMat> inputs;
    for (i32 i = 0; i < input_count; ++i) {
      inputs.emplace_back(
          bytesToImage_gpu(input_columns[0].rows[i].buffer, config_.formats[0]));
    }

    std::vector<cvc::GpuMat> imgs_gray;
    cvc::GpuMat* initial = (cvc::GpuMat*)initial_frame_;
    if (!initial->empty()) {
      cvc::GpuMat gray;
      cvc::cvtColor(*initial, gray, CV_BGR2GRAY);
      imgs_gray.emplace_back(gray);
    } else {
      cudaMemset(output_block, 0, out_buf_size);
      output_columns[0].rows.push_back(Row{output_block, out_buf_size});
      output_block += out_buf_size;
    }

    for (i32 i = 0; i < inputs.size(); ++i) {
      i32 sid = i % num_cuda_streams_;
      cv::cuda::Stream& s = streams_[sid];
      cvc::GpuMat gray;
      cvc::cvtColor(inputs[i], gray, CV_BGR2GRAY, 0, s);
      imgs_gray.emplace_back(gray);
    }

    for (cv::cuda::Stream& s : streams_) {
      s.waitForCompletion();
    }

    cv::Ptr<cvc::DenseOpticalFlow> flow =
      cvc::FarnebackOpticalFlow::create();
      //cvc::OpticalFlowDual_TVL1::create();

    for (i32 i = 0; i < imgs_gray.size() - 1; ++i) {
      u8* output_buf = output_block + i * out_buf_size;
      cvc::GpuMat output_flow_gpu(
        config_.formats[0].height(), config_.formats[0].width(), CV_32FC2, output_buf);
      i32 sid = i % num_cuda_streams_;
      cv::cuda::Stream& s = streams_[sid];

      auto start = now();
      flow->calc(imgs_gray[i], imgs_gray[i+1], output_flow_gpu, s);
      if (profiler_) {
        profiler_->add_interval("flowcalc", start, now());
      }

      output_columns[0].rows.push_back(Row{output_buf, out_buf_size});
    }

    for (cv::cuda::Stream& s : streams_) {
      s.waitForCompletion();
    }

    inputs[inputs.size() - 1].copyTo(*initial);
#else
    LOG(FATAL) << "Cuda not installed.";
#endif  // HAVE_CUDA
  } else {
    std::vector<cvc::GpuMat> inputs;
    for (i32 i = 0; i < input_count; ++i) {
      inputs.emplace_back(
          bytesToImage(input_columns[0].rows[i].buffer, config_.formats[0]));
    }

    cv::Size img_size = inputs[0].size();
#ifdef DEBUG_OPTICAL_FLOW
    i32 out_buf_size = img_size.width * img_size.height * 3;
#else
    i32 out_buf_size = img_size.width * img_size.height * 2 * sizeof(float);
#endif

    std::vector<cv::Mat> imgs_gray;
    cv::Mat* initial = (cv::Mat*)initial_frame_;
    if (!initial->empty()) {
      cv::Mat gray;
      cv::cvtColor(*initial, gray, CV_BGR2GRAY);
      imgs_gray.emplace_back(gray);
    } else {
      u8* out_buf = new u8[out_buf_size];
      std::memset(out_buf, 0, out_buf_size);
      output_columns[0].rows.push_back(Row{out_buf, out_buf_size});
    }

    for (auto& input : inputs) {
      cv::Mat gray;
      cv::cvtColor(input, gray, CV_BGR2GRAY);
      imgs_gray.emplace_back(gray);
    }

    double start = CycleTimer::currentSeconds();

    cv::Ptr<cv::DenseOpticalFlow> flow = cv::createOptFlow_DualTVL1();
    for (i32 i = 0; i < imgs_gray.size() - 1; ++i) {
      cv::Mat output_flow(img_size, CV_32FC2);

      flow->calc(imgs_gray[i], imgs_gray[i + 1], output_flow);

#ifdef DEBUG_OPTICAL_FLOW
      u8* output_buf = new u8[out_buf_size];
      cv::Mat heatmap(img_size, CV_8UC3, output_buf);
      for (int x = 0; x < output_flow.rows; ++x) {
        for (int y = 0; y < output_flow.cols; ++y) {
          cv::Vec2f vel = output_flow.at<cv::Vec2f>(x, y);
          float norm = cv::norm(vel);
          int inorm = std::min((int)std::round(norm * 5), 255);
          heatmap.at<cv::Vec3b>(x, y) = cv::Vec3b(inorm, inorm, inorm);
        }
      }
#else
      u8* output_buf = new u8[out_buf_size];
      std::memcpy(output_buf, output_flow.data, out_buf_size);
#endif

      output_columns[0].rows.push_back(Row{output_buf, out_buf_size});
    }
  }
}

void OpticalFlowEvaluator::set_device() {
  if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
    CU_CHECK(cudaSetDevice(device_id_));
    cvc::setDevice(device_id_);
#endif
  }
}

OpticalFlowEvaluatorFactory::OpticalFlowEvaluatorFactory(DeviceType device_type)
    : device_type_(device_type) {}

EvaluatorCapabilities OpticalFlowEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = device_type_;
  caps.max_devices = device_type_ == DeviceType::GPU
                         ? 1
                         : EvaluatorCapabilities::UnlimitedDevices;
  caps.warmup_size = 1;
  return caps;
}

std::vector<std::string> OpticalFlowEvaluatorFactory::get_output_columns(
  const std::vector<std::string>& input_columns)
{
  return {"opticalflow"};
}

Evaluator* OpticalFlowEvaluatorFactory::new_evaluator(
    const EvaluatorConfig& config) {
  return new OpticalFlowEvaluator(device_type_, config.device_ids[0]);
}
}
