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

#pragma once

#include "scanner/eval/evaluator.h"
#include "scanner/util/common.h"

#include <vector>

namespace scanner {

/**
 * @brief Describes what hardware an evaluator needs to run.
 *
 * Scanner will provide both devices and memory corresponding to
 * an evaluator's capabilities. For example, if device_type = DeviceType::GPU
 * and max_devices = 2 then an evaluator will receive up to two GPU device IDs
 * to run on, and its input and output buffers must be GPU memory.
 */
struct EvaluatorCapabilities {
  static const i32 UnlimitedDevices = 0;

  /** Determines what kind of hardware the evaluator will run on and what kind
   *  of memory it uses. */
  DeviceType device_type;

  /**
   * @brief Scanner will provide up to max_devices for the evaluator to use.
   *
   * If max_devices = EvaluatorCapabilities::UnlimitedDevices, then it will
   * attempt to use all the devices on the machine if possible.
   */
  i32 max_devices;

  /**
   * @brief Specifies how many additional frames to receive per continuous set
   *        of batches.
   *
   * If your stateful evaluator produces problems at discontinuities, e.g.
   * optical flow, then warmup_size allows you to specify a number of frames $k$
   * such that just after a reset, when you would have received $n$ frames from
   * batch $i$, you instead receive $k$ frames from batch $i-1$ and $n-k$ frames
   * batch $i$. Scanner will handle tracking and discarding the first $k$
   * outputs of your evaluator pipeline after a reset.
   */
  i32 warmup_size = 0;

  /**
   * @brief Allows this evaluator to run concurrently with those around it.
   *
   * Currently only works on the first and last evaluators in a pipeline, most
   * commonly the decoder and encoders.
   */
  bool can_overlap = false;
};

/** Information for an evaluator instance on the kinds of input it will receive.
 */
struct EvaluatorConfig {
  DeviceType device_type;
  std::vector<i32> device_ids;
  i32 max_input_count;
  i32 max_frame_width;
  i32 max_frame_height;
};

/**
 * @brief Interface for constructing evaluators at runtime.
 *
 * Scanner pipelines are composed of a sequence of evaluator factories. A single
 * job may use any number of a given evaluator, so the EvaluatorFactory allows
 * the user to capture configuration information about the evaluator (e.g. batch
 * size of a neural net, device type) and pass that information to each new
 * evaluator instance. The EvaluatorFactory also provides metadata about
 * the inputs and outputs from the evaluator it produces.
 */
class EvaluatorFactory {
 public:
  virtual ~EvaluatorFactory(){};

  /** Describes the capabilities of the evaluators the factory produces. */
  virtual EvaluatorCapabilities get_capabilities() = 0;

  /**
   * @brief Returns a list of column names for the evaluator's output.
   *
   * The length of the vector defines the number of columns the evaluator must
   * return, and the string names define the column name. Column names are
   * currently only used when retrieving outputs from disk.
   */
  virtual std::vector<std::string> get_output_names() = 0;

  /* @brief Constructs an evaluator to be used for processing rows of data.
   *
   * This function must be thread-safe but the evaluators created by it do not
   * need to be.
   */
  virtual Evaluator* new_evaluator(const EvaluatorConfig& config) = 0;
};
}
