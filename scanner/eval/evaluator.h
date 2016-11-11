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

#include "scanner/util/common.h"
#include "scanner/util/db.h"
#include "scanner/util/profiler.h"

#include <vector>

namespace scanner {

struct InputFormat {
 public:
  InputFormat() : width_(0), height_(0) {}
  InputFormat(i32 width, i32 height) : width_(width), height_(height) {}

  i32 width() const { return width_; }
  i32 height() const { return height_; }

 private:
  i32 width_;
  i32 height_;
};
/**
 * @brief Interface for a unit of computation in a pipeline.
 *
 * Evaluators form the core of Scanner's interface. They are essentially
 * functions that take rows of inputs and produce an equal number rows of
 * outputs. Evaluators are stateful operators that get reset when provided
 * non-contiguous batches of input. See EvaluatorFactory for how an evaluator
 * defines what hardware it can use for its computation.
 */
class Evaluator {
 public:
  virtual ~Evaluator(){};

  /**
   * @brief Updates the evaluator when running on a new image or video.
   *
   * @param metadata Metadata about the frames the evaluator will receive
   *
   * This provides the evaluator with information about its input like
   * dimensions.
   */
  virtual void configure(const InputFormat& metadata) { metadata_ = metadata; };

  /**
   * @brief Resets evaluators when about to receive non-consecutive inputs.
   *
   * Scanner tries to run evaluators on consecutive blocks of inputs to
   * maximize the accuracy of stateful algorithms like video trackers.
   * However, when the runtime provides an evaluator with a non-consecutive
   * input (because of work imbalance or other reasons), it will call reset
   * to allow the evaluator to reset its state.
   */
  virtual void reset(){};

  /**
   * @brief Runs the evaluator on input rows and produces equal number of
   *        output rows.
   *
   * @param input_buffers
   *        vector of columns, where each column is a vector of inputs and each
   *        input is a byte array
   * @param input_sizes
   *        same structure as input_buffers, and each entry specifies the number
   *        of bytes in the corresponding input byte array
   * @param output_buffers
   *        evaluator output, each column must have same length as the number of
   *        input rows
   * @param output_sizes
   *        same structure as output_buffers, and each entry specifies the
   *        number of bytes in the corresponding output byte array
   *
   * Evaluate gets run on batches of inputs. At the beginning of a pipeline this
   * is raw RGB images from the input images/videos, and after that the input
   * becomes whatever was returned by the previous evaluator.
   *
   * Output buffers must have non-zero size.
   */
  virtual void evaluate(const std::vector<std::vector<u8*>>& input_buffers,
                        const std::vector<std::vector<size_t>>& input_sizes,
                        std::vector<std::vector<u8*>>& output_buffers,
                        std::vector<std::vector<size_t>>& output_sizes) = 0;

  /**
   * Do not call this function.
   */
  void set_profiler(Profiler* profiler) { profiler_ = profiler; }

 protected:
  /**
   * The profiler allows an evaluator to save profiling data for later
   * visualization. It is not guaranteed to be non-null, so check before use.
   */
  Profiler* profiler_ = nullptr;

  /** configure() by default will save the metadata for use in evaluate(). */
  InputFormat metadata_;
};
}
