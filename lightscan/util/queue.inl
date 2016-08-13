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

#include "queue.h"

namespace lightscan {

template <typename T>
Queue<T>::Queue()
  : waiters_(0)
{}

template <typename T>
int Queue<T>::size() {
  std::unique_lock<std::mutex> lock(mutex_);
  return data_.size() - waiters_;
}

template <typename T>
template <typename... Args>
void Queue<T>::emplace(Args&&... args) {
  std::unique_lock<std::mutex> lock(mutex_);
  data_.emplace_back(std::forward<Args>(args)...);
  lock.unlock();
  not_empty_.notify_one();
}

template <typename T>
void Queue<T>::push(T item) {
  std::unique_lock<std::mutex> lock(mutex_);
  data_.push_back(item);
  lock.unlock();
  // TODO(apoms): check how much overhead this causes. Would it be better to
  //              check if the deque was empty before and only notify then
  //              instead of always notifying?
  not_empty_.notify_one();
}

template <typename T>
bool Queue<T>::try_pop(T& item) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (data_.empty()) {
    return false;
  } else {
    item = data_.front();
    data_.pop_front();
    return true;
  }
}

template <typename T>
void Queue<T>::pop(T& item) {
  std::unique_lock<std::mutex> lock(mutex_);
  waiters_++;
  not_empty_.wait(lock, [this]{ return data_.size() > 0; });
  waiters_--;

  item = data_.front();
  data_.pop_front();
}

}
