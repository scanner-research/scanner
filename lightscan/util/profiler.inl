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

#include "lightscan/util/profiler.h"

namespace lightscan {

///////////////////////////////////////////////////////////////////////////////
/// Profiler
inline void Profiler::add_interval(
  const std::string& key,
  timepoint_t start,
  timepoint_t end)
{
  spin_lock();
  records_.emplace_back(TaskRecord{
    key,
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      start - base_time_).count(),
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      end - base_time_).count()});
  unlock();
}

inline void Profiler::spin_lock() {
  while (lock_.test_and_set(std::memory_order_acquire));
}

inline void Profiler::unlock() {
  lock_.clear(std::memory_order_release);
}

}
