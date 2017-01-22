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
#include "scanner/util/profiler.h"

#include <vector>

namespace scanner {

bool is_frame_column(const std::string& name);

struct FrameInfo {
  i32 width;
  i32 height;
};

FrameInfo get_frame_info(const std::string& name);

struct Row {
  u8* buffer;
  size_t size;
};

struct Column {
  std::vector<Row> rows;
};

using BatchedColumns = std::vector<Column>;

/**
 * @brief Interface for a unit of computation in a pipeline.
 *
 * Kernels form the core of Scanner's interface. They are essentially
 * functions that take rows of inputs and produce an equal number rows of
 * outputs. Kernels are stateful operators that get reset when provided
 * non-contiguous batches of input. See KernelFactory for how an evaluator
 * defines what hardware it can use for its computation.
 */
class Kernel {
 public:
  static const i32 UnlimitedDevices = 0;

  struct Config {
    std::vector<DeviceHandle> devices;
    std::vector<std::string> input_columns;
    std::vector<u8> args;
  };

  Kernel(const Config& config);

  virtual ~Kernel(){};

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
   * @param input_columns
   *        vector of columns, where each column is a vector of inputs and each
   *        input is a byte array
   * @param output_columns
   *        evaluator output, each column must have same length as the number of
   *        input rows
   *
   * Evaluate gets run on batches of inputs. At the beginning of a pipeline this
   * is raw RGB images from the input images/videos, and after that the input
   * becomes whatever was returned by the previous evaluator.
   *
   * Number of output columns must be non-zero.
   */
  virtual void execute(const BatchedColumns &input_columns,
                       BatchedColumns &output_columns) = 0;

  /**
   * Do not call this function.
   */
  virtual void set_profiler(Profiler* profiler) { profiler_ = profiler; }

 protected:
  /**
   * The profiler allows an evaluator to save profiling data for later
   * visualization. It is not guaranteed to be non-null, so check before use.
   */
  Profiler* profiler_ = nullptr;
};

#define ROW_BUFFER(column__, row__) (column__.rows[row__].buffer)

#define ROW_SIZE(column__, row__) (column__.rows[row__].size)

#define INSERT_ROW(column__, buffer__, size__) \
  column__.rows.push_back(Row{buffer__, size__})

class KernelBuilder;

using KernelConstructor = std::function<Kernel*(const Kernel::Config& config)>;

class KernelRegistration {
  KernelRegistration(const KernelBuilder& builder);
};

class KernelBuilder {
public:
  friend class KernelRegistration;

  KernelBuilder(const std::string &name,
                KernelConstructor constructor)
    : name_(name), constructor_(constructor) {}

  KernelBuilder& device(DeviceType device_type) {
    device_type_ = device_type;
    return *this;
  }

  KernelBuilder& num_devices(i32 devices) {
    num_devices_ = devices;
    return *this;
  }

 private:
  std::string name_;
  KernelConstructor constructor_;
  DeviceType device_type_;
  i32 num_devices_;
};

#define REGISTER_KERNEL(name, KERNEL) \
  REGISTER_KERNEL_UID(__COUNTER__, name, kernel)

#define REGISTER_KERNEL_UID(uid, name, KERNEL)              \
  static ::scanner::KernelRegistration                      \
    kernel_registration_##uid## =                           \
    KernelBuilder(#name, [](const Kernel::Config &config) { \
        return new KERNEL(config);                          \
      })

}
