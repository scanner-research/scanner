#include "histogram_evaluator.h"

namespace scanner {

const i32 BINS = 16;

void HistogramEvaluator::evaluate(
  std::vector<Mat>& inputs,
  std::vector<u8*>& output_buffers,
  std::vector<size_t>& output_sizes) {
  i64 hist_size = BINS * 3 * sizeof(float);
  for (auto& img_ : inputs) {
    cv::Mat img(img_);

    std::vector<cv::Mat> bgr_planes;
    cv::split(img, bgr_planes);

    cv::Mat r_hist, g_hist, b_hist;
    float range[] = {0, 256};
    const float* histRange = {range};

    cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &BINS, &histRange);
    cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &BINS, &histRange);
    cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &BINS, &histRange);

    std::vector<cv::Mat> hists = {r_hist, g_hist, b_hist};
    cv::Mat hist;
    cv::hconcat(hists, hist);

    #ifdef HAVE_CUDA
    u8* hist_buffer;
    cudaMalloc((void**) &hist_buffer, hist_size);
    cudaMemcpy(hist_buffer, img.data, hist_size, cudaMemcpyHostToDevice);
    #else
    u8* hist_buffer = new u8[hist_size];
    assert(hist_size == hist.total() * hist.elemSize());
    memcpy(hist_buffer, hist.data, hist_size);
    #endif

    output_sizes.push_back(hist_size);
    output_buffers.push_back(hist_buffer);
  }
}

}
