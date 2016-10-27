#include "optical_flow_evaluator.h"

#include <cstring>
#include <opencv2/video.hpp>
#include "scanner/util/cycle_timer.h"

//#define USE_OFDIS

#ifdef USE_OFDIS
#include "oflow.h"
#endif

namespace scanner {

#ifdef USE_OFDIS

int AutoFirstScaleSelect(int imgwidth, int fratio, int patchsize) {
  return std::max(0, (int)std::floor(log2((2.0f * (float)imgwidth) /
                                          ((float)fratio * (float)patchsize))));
}

void ConstructImgPyramide(const cv::Mat& img_ao_fmat, cv::Mat* img_ao_fmat_pyr,
                          cv::Mat* img_ao_dx_fmat_pyr,
                          cv::Mat* img_ao_dy_fmat_pyr, const float** img_ao_pyr,
                          const float** img_ao_dx_pyr,
                          const float** img_ao_dy_pyr, const int lv_f,
                          const int lv_l, const int rpyrtype,
                          const bool getgrad, const int imgpadding,
                          const int padw, const int padh) {
  for (int i = 0; i <= lv_f; ++i)  // Construct image and gradient pyramides
  {
    if (i == 0)  // At finest scale: copy directly, for all other: downscale
                 // previous scale by .5
    {
#if (SELECTCHANNEL == 1 | \
     SELECTCHANNEL == 3)  // use RGB or intensity image directly
      img_ao_fmat_pyr[i] = img_ao_fmat.clone();
#elif (SELECTCHANNEL == 2)  // use gradient magnitude image as input
      cv::Mat dx, dy, dx2, dy2, dmag;
      cv::Sobel(img_ao_fmat, dx, CV_32F, 1, 0, 1, 1, 0, cv::BORDER_DEFAULT);
      cv::Sobel(img_ao_fmat, dy, CV_32F, 0, 1, 1, 1, 0, cv::BORDER_DEFAULT);
      dx2 = dx.mul(dx);
      dy2 = dy.mul(dy);
      dmag = dx2 + dy2;
      cv::sqrt(dmag, dmag);
      img_ao_fmat_pyr[i] = dmag.clone();
#endif
    } else {
      cv::resize(img_ao_fmat_pyr[i - 1], img_ao_fmat_pyr[i], cv::Size(), .5, .5,
                 cv::INTER_LINEAR);
    }

    img_ao_fmat_pyr[i].convertTo(img_ao_fmat_pyr[i], rpyrtype);

    if (getgrad) {
      cv::Sobel(img_ao_fmat_pyr[i], img_ao_dx_fmat_pyr[i], CV_32F, 1, 0, 1, 1,
                0, cv::BORDER_DEFAULT);
      cv::Sobel(img_ao_fmat_pyr[i], img_ao_dy_fmat_pyr[i], CV_32F, 0, 1, 1, 1,
                0, cv::BORDER_DEFAULT);
      img_ao_dx_fmat_pyr[i].convertTo(img_ao_dx_fmat_pyr[i], CV_32F);
      img_ao_dy_fmat_pyr[i].convertTo(img_ao_dy_fmat_pyr[i], CV_32F);
    }
  }

  // pad images
  for (int i = 0; i <= lv_f; ++i)  // Construct image and gradient pyramides
  {
    cv::copyMakeBorder(
        img_ao_fmat_pyr[i], img_ao_fmat_pyr[i], imgpadding, imgpadding,
        imgpadding, imgpadding,
        cv::BORDER_REPLICATE);  // Replicate border for image padding
    img_ao_pyr[i] = (float*)img_ao_fmat_pyr[i].data;

    if (getgrad) {
      cv::copyMakeBorder(img_ao_dx_fmat_pyr[i], img_ao_dx_fmat_pyr[i],
                         imgpadding, imgpadding, imgpadding, imgpadding,
                         cv::BORDER_CONSTANT, 0);  // Zero padding for gradients
      cv::copyMakeBorder(img_ao_dy_fmat_pyr[i], img_ao_dy_fmat_pyr[i],
                         imgpadding, imgpadding, imgpadding, imgpadding,
                         cv::BORDER_CONSTANT, 0);

      img_ao_dx_pyr[i] = (float*)img_ao_dx_fmat_pyr[i].data;
      img_ao_dy_pyr[i] = (float*)img_ao_dy_fmat_pyr[i].data;
    }
  }
}

// Only works with OpenCV 2.4.x
void ofdis(const cv::Mat& img_a, const cv::Mat& img_b, cv::Mat& output) {
  cv::Mat img_ao_mat, img_bo_mat;
  img_a.copyTo(img_ao_mat);
  img_b.copyTo(img_bo_mat);
  cv::Size sz = img_ao_mat.size();
  int width_org = sz.width;
  int height_org = sz.height;
  int fratio = 5;
  int nochannels = 1;
  int rpyrtype = CV_32FC1;

  int lv_f, lv_l, maxiter, miniter, patchsz, patnorm, costfct, tv_innerit,
      tv_solverit;
  float mindprate, mindrrate, minimgerr, poverl, tv_alpha, tv_gamma, tv_delta,
      tv_sor;
  bool usefbcon, usetvref;

  mindprate = 0.05;
  mindrrate = 0.95;
  minimgerr = 0.0;
  usefbcon = 0;
  patnorm = 1;
  costfct = 0;
  tv_alpha = 10.0;
  tv_gamma = 10.0;
  tv_delta = 5.0;
  tv_innerit = 1;
  tv_solverit = 3;
  tv_sor = 1.6;
  patchsz = 8;
  poverl = 0.4;
  lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
  lv_l = std::max(lv_f - 2, 0);
  maxiter = 12;
  miniter = 12;
  usetvref = 1;

  int padw = 0, padh = 0;
  int scfct = pow(2, lv_f);
  int div = sz.width % scfct;
  if (div > 0) padw = scfct - div;
  div = sz.height % scfct;
  if (div > 0) padh = scfct - div;
  if (padh > 0 || padw > 0) {
    cv::copyMakeBorder(img_ao_mat, img_ao_mat, floor((float)padh / 2.0f),
                       ceil((float)padh / 2.0f), floor((float)padw / 2.0f),
                       ceil((float)padw / 2.0f), cv::BORDER_REPLICATE);
    cv::copyMakeBorder(img_bo_mat, img_bo_mat, floor((float)padh / 2.0f),
                       ceil((float)padh / 2.0f), floor((float)padw / 2.0f),
                       ceil((float)padw / 2.0f), cv::BORDER_REPLICATE);
  }
  sz = img_ao_mat.size();  // padded image size, ensures divisibility by 2 on
                           // all scales (except last)

  cv::Mat img_ao_fmat, img_bo_fmat;
  img_ao_mat.convertTo(img_ao_fmat, CV_32F);
  img_bo_mat.convertTo(img_bo_fmat, CV_32F);

  const float* img_ao_pyr[lv_f + 1];
  const float* img_bo_pyr[lv_f + 1];
  const float* img_ao_dx_pyr[lv_f + 1];
  const float* img_ao_dy_pyr[lv_f + 1];
  const float* img_bo_dx_pyr[lv_f + 1];
  const float* img_bo_dy_pyr[lv_f + 1];

  cv::Mat img_ao_fmat_pyr[lv_f + 1];
  cv::Mat img_bo_fmat_pyr[lv_f + 1];
  cv::Mat img_ao_dx_fmat_pyr[lv_f + 1];
  cv::Mat img_ao_dy_fmat_pyr[lv_f + 1];
  cv::Mat img_bo_dx_fmat_pyr[lv_f + 1];
  cv::Mat img_bo_dy_fmat_pyr[lv_f + 1];

  ConstructImgPyramide(img_ao_fmat, img_ao_fmat_pyr, img_ao_dx_fmat_pyr,
                       img_ao_dy_fmat_pyr, img_ao_pyr, img_ao_dx_pyr,
                       img_ao_dy_pyr, lv_f, lv_l, rpyrtype, 1, patchsz, padw,
                       padh);
  ConstructImgPyramide(img_bo_fmat, img_bo_fmat_pyr, img_bo_dx_fmat_pyr,
                       img_bo_dy_fmat_pyr, img_bo_pyr, img_bo_dx_pyr,
                       img_bo_dy_pyr, lv_f, lv_l, rpyrtype, 1, patchsz, padw,
                       padh);

  float sc_fct = pow(2, lv_l);
  cv::Mat flowout(sz.height / sc_fct, sz.width / sc_fct, CV_32FC2);

  OFC::OFClass ofc(img_ao_pyr, img_ao_dx_pyr, img_ao_dy_pyr, img_bo_pyr,
                   img_bo_dx_pyr, img_bo_dy_pyr, patchsz, (float*)flowout.data,
                   nullptr, sz.width, sz.height, lv_f, lv_l, maxiter, miniter,
                   mindprate, mindrrate, minimgerr, patchsz, poverl, usefbcon,
                   costfct, nochannels, patnorm, usetvref, tv_alpha, tv_gamma,
                   tv_delta, tv_innerit, tv_solverit, tv_sor, 0);

  if (lv_l != 0) {
    flowout *= sc_fct;
    cv::resize(flowout, flowout, cv::Size(), sc_fct, sc_fct, cv::INTER_LINEAR);
  }

  flowout =
      flowout(cv::Rect((int)floor((float)padw / 2.0f),
                       (int)floor((float)padh / 2.0f), width_org, height_org));
  flowout.copyTo(output);
}
#endif

OpticalFlowEvaluator::OpticalFlowEvaluator(DeviceType device_type)
    : device_type_(device_type) {
  reset();
}

OpticalFlowEvaluator::~OpticalFlowEvaluator() {
  if (initial_frame != nullptr) {
    if (device_type_ == DeviceType::GPU) {
      delete ((cvc::GpuMat*)initial_frame_);
    } else {
      delete ((cv::Mat*)initial_frame_);
    }
  }
}

void OpticalFlowEvaluator::reset() {
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

void OpticalFlowEvaluator::evaluate(
    const std::vector<std::vector<u8*>>& input_buffers,
    const std::vector<std::vector<size_t>>& input_sizes,
    std::vector<std::vector<u8*>>& output_buffers,
    std::vector<std::vector<size_t>>& output_sizes) {
  if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
    std::vector<cvc::GpuMat> inputs;
    for (i32 i = 0; i < input_buffers[0].size(); ++i) {
      inputs.emplace_back(bytesToImage_gpu(input_buffers[0][i], metadata_));
    }

    cv::Size img_size = inputs[0].size();
    i32 out_buf_size = img_size.width * img_size.height * 2 * sizeof(float);

    std::vector<cvc::GpuMat> imgs_gray;
    cvc::GpuMat* initial = (cvc::GpuMat*)initial_frame_;
    if (!initial->empty()) {
      cvc::GpuMat gray;
      cvc::cvtColor(*initial, gray, CV_BGR2GRAY);
      imgs_gray.emplace_back(gray);
    } else {
      u8* out_buf;
      cudaMalloc((void**)&out_buf, out_buf_size);
      cudaMemset(out_buf, 0, out_buf_size);
      output_buffers[0].push_back(out_buf);
      output_sizes[0].push_back(out_buf_size);
    }

    for (auto& input : inputs) {
      cvc::GpuMat gray;
      cvc::cvtColor(input, gray, CV_BGR2GRAY);
      imgs_gray.emplace_back(gray);
    }

    cv::Ptr<cvc::DenseOpticalFlow> flow = cvc::OpticalFlowDual_TVL1::create();

    for (i32 i = 0; i < imgs_gray.size() - 1; ++i) {
      u8* output_buf;
      cudaMalloc((void**)&output_buf, out_buf_size);
      cvc::GpuMat output_flow_gpu(img_size, CV_32FC2, output_buf);

      flow->calc(imgs_gray[i], imgs_gray[i + 1], output_flow_gpu);

      // cv::Mat output_flow(output_flow_gpu);

      // u8* heatmap_buf = new u8[out_buf_size];
      // cv::Mat heatmap(img_size, CV_8UC3, heatmap_buf);
      // for (int x = 0; x < output_flow.rows; ++x) {
      //   for (int y = 0; y < output_flow.cols; ++y) {
      //     cv::Vec2f vel = output_flow.at<cv::Vec2f>(x, y);
      //     float norm = cv::norm(vel);
      //     int inorm = std::min((int) std::round(norm * 100), 255);
      //     heatmap.at<cv::Vec3b>(x, y) = cv::Vec3b(inorm, inorm, inorm);
      //   }
      // }

      // u8* output_buf;
      // cudaMalloc((void**) &output_buf, out_buf_size);
      // cudaMemcpy(output_buf, heatmap_buf, out_buf_size,
      // cudaMemcpyHostToDevice);

      output_sizes[0].push_back(out_buf_size);
      output_buffers[0].push_back(output_buf);
    }

    inputs[inputs.size() - 1].copyTo(*initial);
#else
    LOG(FATAL) << "Cuda not installed.";
#endif  // HAVE_CUDA
  } else {
    std::vector<cvc::GpuMat> inputs;
    for (i32 i = 0; i < input_buffers[0].size(); ++i) {
      inputs.emplace_back(bytesToImage(input_buffers[0][i], metadata_));
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
      output_buffers[0].push_back(out_buf);
      output_sizes[0].push_back(out_buf_size);
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

      // ofdis(imgs_gray[i], imgs_gray[i+1], output_flow);
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

      output_sizes[0].push_back(out_buf_size);
      output_buffers[0].push_back(output_buf);
    }
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

std::vector<std::string> OpticalFlowEvaluatorFactory::get_output_names() {
  return {"opticalflow"};
}

Evaluator* OpticalFlowEvaluatorFactory::new_evaluator(
    const EvaluatorConfig& config) {
  return new OpticalFlowEvaluator(device_type_);
}
}
