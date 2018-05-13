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

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>

#include "scanner/util/blockingconcurrentqueue.h"

namespace scanner {

using namespace moodycamel;

template <typename T>
class Queue : public BlockingConcurrentQueue<T> {
 public:
  Queue(size_t size=8) : BlockingConcurrentQueue<T>(size) {}

  inline void clear() {
    T t;
    while (BlockingConcurrentQueue<T>::try_dequeue(t)) {}
  }

  inline size_t size() {
    return BlockingConcurrentQueue<T>::size_approx();
  }

  inline void push(T item) {
    bool success = BlockingConcurrentQueue<T>::enqueue(item);
    LOG_IF(FATAL, !success) << "Queue push failed";
  }

  inline void pop(T& item) {
    BlockingConcurrentQueue<T>::wait_dequeue(item);
  }
};

}
