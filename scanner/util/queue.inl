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

namespace scanner {

template <typename T>
Queue<T>::Queue(i32 max_size)
    : max_size_(max_size) {}

template <typename T>
Queue<T>::Queue(Queue<T> &&o)
    : max_size_(o.max_size_), data_(std::move(o.data_)) {}

template <typename T>
int Queue<T>::size() {
  std::unique_lock<std::mutex> lock(mutex_);
  return data_.size() - pop_waiters_ + push_waiters_;
}

template <typename T>
template <typename... Args>
void Queue<T>::emplace(Args&&... args) {
  std::unique_lock<std::mutex> lock(mutex_);
  push_waiters_++;
  not_full_.wait(lock, [this]{ return data_.size() < max_size_; });
  push_waiters_--;
  data_.emplace_back(std::forward<Args>(args)...);

  lock.unlock();
  not_empty_.notify_one();
}

template <typename T>
void Queue<T>::push(T item) {
  std::unique_lock<std::mutex> lock(mutex_);
  push_waiters_++;
  not_full_.wait(lock, [this]{ return data_.size() < max_size_; });
  push_waiters_--;

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

    lock.unlock();
    not_full_.notify_one();
    return true;
  }
}

template <typename T>
void Queue<T>::pop(T& item) {
  std::unique_lock<std::mutex> lock(mutex_);
  pop_waiters_++;
  not_empty_.wait(lock, [this]{ return data_.size() > 0; });
  pop_waiters_--;

  item = data_.front();
  data_.pop_front();

  lock.unlock();
  not_full_.notify_one();
}

template <typename T>
void Queue<T>::peek(T& item) {
  std::unique_lock<std::mutex> lock(mutex_);
  pop_waiters_++;
  not_empty_.wait(lock, [this]{ return data_.size() > 0; });
  pop_waiters_--;

  item = data_.front();

  lock.unlock();
}

template <typename T>
void Queue<T>::clear() {
  std::unique_lock<std::mutex> lock(mutex_);
  data_.clear();

  lock.unlock();
  not_full_.notify_one();
}

}
