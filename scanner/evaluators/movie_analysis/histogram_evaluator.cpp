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

const int BINS = 256;

HistogramEvaluator::HistogramEvaluator(EvaluatorConfig) {}

HistogramEvaluator::~HistogramEvaluator() {}

void HistogramEvaluator::configure(const DatasetItemMetadata& metadata) {
  this->metadata = metadata;
}

void HistogramEvaluator::evaluate(
  u8* input_buffer,
  std::vector<std::vector<u8*>>& output_buffers,
  std::vector<std::vector<size_t>>& output_sizes,
  i32 batch_size)
{
  i64 frame_size = metadata.width * metadata.height * 3 * sizeof(u8);
  i64 hist_size = BINS * 3 * sizeof(u8);
  for (i32 i = 0; i < batch_size; i++) {
    u8* frame_buffer = (u8*)(input_buffer + frame_size * i);
    u8* hist_buffer = new u8[hist_size];
    cv::Mat img(metadata.height, metadata.width, CV_8UC3, frame_buffer);
    std::vector<cv::Mat> bgr_planes;
    cv::split(img, bgr_planes);

    cv::Mat r_hist, g_hist, b_hist;
    float range[] = {0, 256};
    const float* histRange = { range };

    cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &BINS, &histRange);
    cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &BINS, &histRange);
    cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &BINS, &histRange);

    std::vector<cv::Mat> hists = { r_hist, g_hist, b_hist };
    cv::Mat hist, hist_u8;
    cv::hconcat(hists, hist);
    hist.convertTo(hist_u8, CV_8U);

    memcpy(hist_buffer, hist_u8.data, hist_u8.total() * hist_u8.elemSize());

    output_sizes[0].push_back(hist_size);
    output_buffers[0].push_back(hist_buffer);
  }
}

HistogramEvaluatorConstructor::HistogramEvaluatorConstructor() {}

HistogramEvaluatorConstructor::~HistogramEvaluatorConstructor() {}

int HistogramEvaluatorConstructor::get_number_of_devices() {
  return 1;
}

DeviceType HistogramEvaluatorConstructor::get_input_buffer_type() {
  return DeviceType::CPU;
}

DeviceType HistogramEvaluatorConstructor::get_output_buffer_type() {
  return DeviceType::CPU;
}

int HistogramEvaluatorConstructor::get_number_of_outputs() {
  return 1;
}

std::vector<std::string> HistogramEvaluatorConstructor::get_output_names() {
  return {"histogram"};
}

u8*
HistogramEvaluatorConstructor::new_input_buffer(const EvaluatorConfig& config) {
  return new u8[
    config.max_batch_size *
    config.max_frame_width *
    config.max_frame_height *
    3 *
    sizeof(u8)];
}

void HistogramEvaluatorConstructor::delete_input_buffer(
  const EvaluatorConfig& config,
  u8* buffer)
{
  delete[] buffer;
}

void HistogramEvaluatorConstructor::delete_output_buffer(
  const EvaluatorConfig& config,
  u8* buffer)
{
  delete[] buffer;
}

Evaluator*
HistogramEvaluatorConstructor::new_evaluator(const EvaluatorConfig& config) {
  return new HistogramEvaluator(config);
}

}
