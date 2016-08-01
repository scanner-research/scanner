/* Copyright 2016 Carnegie Mellon University, NVIDIA Corporation
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

#include "lightscan/util/common.h"

#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <atomic>

namespace lightscan {

class Profiler {
public:
  Profiler(timepoint_t base_time);

  Profiler(const Profiler& other);

  ~Profiler(void);

  void add_interval(
    const std::string& key,
    timepoint_t start,
    timepoint_t end);

  struct TaskRecord {
    std::string key;
    int64_t start;
    int64_t end;
  };

  const std::vector<TaskRecord>& get_records() const;

protected:
  void spin_lock();
  void unlock();

  timepoint_t base_time_;
  std::atomic_flag lock_;
  std::vector<TaskRecord> records_;
};

} // namespace lightscan

#include "lightscan/util/profiler.inl"
