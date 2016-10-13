#include "optical_flow_evaluator.h"
#include <opencv2/video.hpp>

#define USE_OFDIS

#ifdef USE_OFDIS
#define SELECTMODE 1
#include "oflow.h"
#endif

namespace scanner {

int AutoFirstScaleSelect(int imgwidth, int fratio, int patchsize)
{
  return std::max(0,(int)std::floor(log2((2.0f*(float)imgwidth) / ((float)fratio * (float)patchsize))));
}

void ConstructImgPyramide(const cv::Mat & img_ao_fmat, cv::Mat * img_ao_fmat_pyr, cv::Mat * img_ao_dx_fmat_pyr, cv::Mat * img_ao_dy_fmat_pyr, const float ** img_ao_pyr, const float ** img_ao_dx_pyr, const float ** img_ao_dy_pyr, const int lv_f, const int lv_l, const int rpyrtype, const bool getgrad, const int imgpadding, const int padw, const int padh)
{
  for (int i=0; i<=lv_f; ++i)  // Construct image and gradient pyramides
  {
    if (i==0) // At finest scale: copy directly, for all other: downscale previous scale by .5
    {
#if (SELECTCHANNEL==1 | SELECTCHANNEL==3)  // use RGB or intensity image directly
      img_ao_fmat_pyr[i] = img_ao_fmat.clone();
#elif (SELECTCHANNEL==2)   // use gradient magnitude image as input
      cv::Mat dx,dy,dx2,dy2,dmag;
      cv::Sobel( img_ao_fmat, dx, CV_32F, 1, 0, 1, 1, 0, cv::BORDER_DEFAULT );
      cv::Sobel( img_ao_fmat, dy, CV_32F, 0, 1, 1, 1, 0, cv::BORDER_DEFAULT );
      dx2 = dx.mul(dx);
      dy2 = dy.mul(dy);
      dmag = dx2+dy2;
      cv::sqrt(dmag,dmag);
      img_ao_fmat_pyr[i] = dmag.clone();
#endif
    }
    else
      cv::resize(img_ao_fmat_pyr[i-1], img_ao_fmat_pyr[i], cv::Size(), .5, .5, cv::INTER_LINEAR);

    img_ao_fmat_pyr[i].convertTo(img_ao_fmat_pyr[i], rpyrtype);

    if ( getgrad )
    {
      cv::Sobel( img_ao_fmat_pyr[i], img_ao_dx_fmat_pyr[i], CV_32F, 1, 0, 1, 1, 0, cv::BORDER_DEFAULT );
      cv::Sobel( img_ao_fmat_pyr[i], img_ao_dy_fmat_pyr[i], CV_32F, 0, 1, 1, 1, 0, cv::BORDER_DEFAULT );
      img_ao_dx_fmat_pyr[i].convertTo(img_ao_dx_fmat_pyr[i], CV_32F);
      img_ao_dy_fmat_pyr[i].convertTo(img_ao_dy_fmat_pyr[i], CV_32F);
    }
  }

  // pad images
  for (int i=0; i<=lv_f; ++i)  // Construct image and gradient pyramides
  {
    cv::copyMakeBorder(img_ao_fmat_pyr[i],img_ao_fmat_pyr[i],imgpadding,imgpadding,imgpadding,imgpadding,cv::BORDER_REPLICATE);  // Replicate border for image padding
    img_ao_pyr[i] = (float*)img_ao_fmat_pyr[i].data;

    if ( getgrad )
    {
      cv::copyMakeBorder(img_ao_dx_fmat_pyr[i],img_ao_dx_fmat_pyr[i],imgpadding,imgpadding,imgpadding,imgpadding,cv::BORDER_CONSTANT , 0); // Zero padding for gradients
      cv::copyMakeBorder(img_ao_dy_fmat_pyr[i],img_ao_dy_fmat_pyr[i],imgpadding,imgpadding,imgpadding,imgpadding,cv::BORDER_CONSTANT , 0);

      img_ao_dx_pyr[i] = (float*)img_ao_dx_fmat_pyr[i].data;
      img_ao_dy_pyr[i] = (float*)img_ao_dy_fmat_pyr[i].data;
    }
  }
}

void ofdis(cv::Mat& img_ao_mat, cv::Mat& img_bo_mat) {
  cv::Size sz = img_ao_mat.size();
  int width_org = sz.width;
  int height_org = sz.height;
  int fratio = 5;
  int nochannels = 1;
  int rpyrtype = CV_32FC1;

  int lv_f, lv_l, maxiter, miniter, patchsz, patnorm, costfct, tv_innerit, tv_solverit;
  float mindprate, mindrrate, minimgerr, poverl, tv_alpha, tv_gamma, tv_delta, tv_sor;
  bool usefbcon, usetvref;

  mindprate = 0.05; mindrrate = 0.95; minimgerr = 0.0;
  usefbcon = 0; patnorm = 1; costfct = 0;
  tv_alpha = 10.0; tv_gamma = 10.0; tv_delta = 5.0;
  tv_innerit = 1; tv_solverit = 3; tv_sor = 1.6;
  patchsz = 8; poverl = 0.4;
  lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
  lv_l = std::max(lv_f-2,0); maxiter = 12; miniter = 12;
  usetvref = 1;

  int padw=0, padh=0;
  int scfct = pow(2,lv_f);
  int div = sz.width % scfct;
  if (div>0) padw = scfct - div;
  div = sz.height % scfct;
  if (div>0) padh = scfct - div;
  if (padh>0 || padw>0)
  {
    cv::copyMakeBorder(img_ao_mat,img_ao_mat,floor((float)padh/2.0f),ceil((float)padh/2.0f),floor((float)padw/2.0f),ceil((float)padw/2.0f),cv::BORDER_REPLICATE);
    cv::copyMakeBorder(img_bo_mat,img_bo_mat,floor((float)padh/2.0f),ceil((float)padh/2.0f),floor((float)padw/2.0f),ceil((float)padw/2.0f),cv::BORDER_REPLICATE);
  }
  sz = img_ao_mat.size();  // padded image size, ensures divisibility by 2 on all scales (except last)

  cv::Mat img_ao_fmat, img_bo_fmat;
  img_ao_mat.convertTo(img_ao_fmat, CV_32F);
  img_bo_mat.convertTo(img_bo_fmat, CV_32F);

  const float* img_ao_pyr[lv_f+1];
  const float* img_bo_pyr[lv_f+1];
  const float* img_ao_dx_pyr[lv_f+1];
  const float* img_ao_dy_pyr[lv_f+1];
  const float* img_bo_dx_pyr[lv_f+1];
  const float* img_bo_dy_pyr[lv_f+1];

  cv::Mat img_ao_fmat_pyr[lv_f+1];
  cv::Mat img_bo_fmat_pyr[lv_f+1];
  cv::Mat img_ao_dx_fmat_pyr[lv_f+1];
  cv::Mat img_ao_dy_fmat_pyr[lv_f+1];
  cv::Mat img_bo_dx_fmat_pyr[lv_f+1];
  cv::Mat img_bo_dy_fmat_pyr[lv_f+1];

  ConstructImgPyramide(img_ao_fmat, img_ao_fmat_pyr, img_ao_dx_fmat_pyr, img_ao_dy_fmat_pyr, img_ao_pyr, img_ao_dx_pyr, img_ao_dy_pyr, lv_f, lv_l, rpyrtype, 1, patchsz, padw, padh);
  ConstructImgPyramide(img_bo_fmat, img_bo_fmat_pyr, img_bo_dx_fmat_pyr, img_bo_dy_fmat_pyr, img_bo_pyr, img_bo_dx_pyr, img_bo_dy_pyr, lv_f, lv_l, rpyrtype, 1, patchsz, padw, padh);

  float sc_fct = pow(2,lv_l);
  cv::Mat flowout(sz.height / sc_fct , sz.width / sc_fct, CV_32FC2);

  OFC::OFClass ofc(img_ao_pyr, img_ao_dx_pyr, img_ao_dy_pyr,
                   img_bo_pyr, img_bo_dx_pyr, img_bo_dy_pyr,
                   patchsz,
                   (float*)flowout.data,
                   nullptr,
                   sz.width, sz.height,
                   lv_f, lv_l, maxiter, miniter, mindprate, mindrrate, minimgerr, patchsz, poverl,
                   usefbcon, costfct, nochannels, patnorm,
                   usetvref, tv_alpha, tv_gamma, tv_delta, tv_innerit, tv_solverit, tv_sor,
                   0);

  if (lv_l != 0) {
    flowout *= sc_fct;
    cv::resize(flowout, flowout, cv::Size(), sc_fct, sc_fct , cv::INTER_LINEAR);
  }

  flowout = flowout(cv::Rect((int)floor((float)padw/2.0f),(int)floor((float)padh/2.0f),width_org,height_org));

  LOG(INFO) << flowout.size();
  exit(0);
}

void OpticalFlowEvaluator::evaluate(
  std::vector<cv::Mat>& inputs,
  std::vector<u8*>& output_buffers,
  std::vector<size_t>& output_sizes) {

  std::vector<cv::Mat> imgs_gray;
  for (auto& input : inputs) {
    cv::Mat gray;
    cv::cvtColor(input, gray, CV_BGR2GRAY);
    imgs_gray.emplace_back(gray);
  }

#ifdef DEBUG_OPTICAL_FLOW
  i32 img_buf_size = inputs[0].total() * inputs[0].elemSize();
  output_sizes.push_back(img_buf_size);
  u8* first_img = new u8[img_buf_size];
  std::memset(first_img, 0, img_buf_size);
  output_buffers.push_back(first_img);
#else
  output_sizes.push_back(0);
  output_buffers.push_back(new u8[1]);
#endif

  cv::Ptr<cv::DenseOpticalFlow> flow = cv::createOptFlow_DualTVL1();
  cv::Size img_size = inputs[0].size();
  i32 buf_size = img_size.width * img_size.height * 2 * sizeof(float);
  for (i32 i = 0; i < inputs.size() - 1; ++i) {
    u8* output_buf = new u8[buf_size];
    cv::Mat output_flow(img_size, CV_32FC2, output_buf);

    ofdis(imgs_gray[i], imgs_gray[i+1]);

    double start = CycleTimer::currentSeconds();
    flow->calc(imgs_gray[i], imgs_gray[i+1], output_flow);
    LOG(INFO) << CycleTimer::currentSeconds() - start;



#ifdef DEBUG_OPTICAL_FLOW
    u8* output_heatmap = new u8[img_buf_size];
    cv::Mat heatmap(img_size, CV_8UC3, output_heatmap);
    for (int x = 0; x < output_flow.rows; ++x) {
      for (int y = 0; y < output_flow.cols; ++y) {
        cv::Vec2f vel = output_flow.at<cv::Vec2f>(x, y);
        float norm = cv::norm(vel);
        int inorm = std::min((int) std::round(norm * 5), 255);
        heatmap.at<cv::Vec3b>(x, y) = cv::Vec3b(inorm, inorm, inorm);
      }
    }

    output_sizes.push_back(img_buf_size);
    output_buffers.push_back(output_heatmap);
#else
    output_sizes.push_back(buf_size);
    output_buffers.push_back(output_buf);
#endif
  }
}

}
