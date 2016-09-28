/* Copyright 2016 Carnegie Mellon University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "scanner/evaluators/movie_analysis/histogram_evaluator.h"
#include "scanner/util/opencv.h"

namespace scanner {

const i32 BINS = 256;

HistogramEvaluator::HistogramEvaluator(EvaluatorConfig) {}

void HistogramEvaluator::configure(const DatasetItemMetadata& metadata) {
  this->metadata = metadata;
}

void HistogramEvaluator::evaluate(
  i32 input_count, u8* input_buffer,
  std::vector<std::vector<u8*>>& output_buffers,
  std::vector<std::vector<size_t>>& output_sizes)
{
  i64 hist_size = BINS * 3 * sizeof(u8);
  for (i32 i = 0; i < input_count; i++) {
    cv::Mat img = bytesToImage(input_buffer, i, metadata);
    std::vector<cv::Mat> bgr_planes;
    cv::split(img, bgr_planes);

    cv::Mat r_hist, g_hist, b_hist;
    float range[] = {0, 256};
    const float* histRange = {range};

    cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &BINS, &histRange);
    cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &BINS, &histRange);
    cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &BINS, &histRange);

    std::vector<cv::Mat> hists = {r_hist, g_hist, b_hist};
    cv::Mat hist, hist_u8;
    cv::hconcat(hists, hist);
    hist.convertTo(hist_u8, CV_8U);

    u8* hist_buffer = new u8[hist_size];
    memcpy(hist_buffer, hist_u8.data, hist_u8.total() * hist_u8.elemSize());

    output_sizes[0].push_back(hist_size);
    output_buffers[0].push_back(hist_buffer);
  }
}

HistogramEvaluatorFactory::HistogramEvaluatorFactory() {}

EvaluatorCapabilities HistogramEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = DeviceType::CPU;
  caps.device_usage = EvaluatorCapabilities::Single;
  caps.max_devices = 1;
  caps.warmup_size = 0;
  return caps;
}

i32 HistogramEvaluatorFactory::get_number_of_outputs() { return 1; }

std::vector<std::string> HistogramEvaluatorFactory::get_output_names() {
  return {"histogram"};
}

Evaluator* HistogramEvaluatorFactory::new_evaluator(
  const EvaluatorConfig& config) {
  return new HistogramEvaluator(config);
}
}
