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

#include <glog/logging.h>
#include <libgen.h>
#include <stdio.h>
#include <sys/stat.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sys/prctl.h>

namespace scanner {

class SpinLock {
  std::atomic_flag locked = ATOMIC_FLAG_INIT;

 public:
  void lock() {
    while (locked.test_and_set(std::memory_order_acquire)) {
      ;
    }
  }
  void unlock() { locked.clear(std::memory_order_release); }
};

using timepoint_t = std::chrono::time_point<std::chrono::high_resolution_clock>;

inline timepoint_t now() { return std::chrono::high_resolution_clock::now(); }

inline double nano_since(timepoint_t then) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now() - then)
      .count();
}

///////////////////////////////////////////////////////////////////////////////
/// String processing
inline void split(const std::string& s, char delim,
                  std::vector<std::string>& elems) {
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
}

inline std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, elems);
  return elems;
}

///////////////////////////////////////////////////////////////////////////////
/// pthread utils
#define THREAD_RETURN_SUCCESS()      \
  do {                               \
    void* val = malloc(sizeof(int)); \
    *((int*)val) = EXIT_SUCCESS;     \
    pthread_exit(val);               \
  } while (0);

template <typename T>
T sum(const std::vector<T>& vec) {
  T result{};
  for (const T& v : vec) {
    result += v;
  }
  return result;
}

template <typename T>
T nano_to_ms(T ns) {
  return ns / 1000000;
}

class Flag {
 public:
  void set() {
    std::unique_lock<std::mutex> lock(m_);
    bit_ = true;
    lock.unlock();
    cv_.notify_all();
  }

  bool raised() {
    return bit_.load();
  }

  void wait() {
    std::unique_lock<std::mutex> lock(m_);
    cv_.wait(lock, [&] { return bit_.load(); });
  }

  void wait_for(int ms) {
    std::unique_lock<std::mutex> lock(m_);
    cv_.wait_for(lock, std::chrono::milliseconds(ms),
                 [&] { return bit_.load(); });
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  std::atomic<bool> bit_{false};
};
///////////////////////////////////////////////////////////////////////////////
/// Debugging utils

// Hacky way to print a stack trace while running. Useful right before
// a LOG(FATAL) or other type of fatal event.
inline void print_trace() {
  char pid_buf[30];
  sprintf(pid_buf, "%d", getpid());
  char name_buf[512];
  name_buf[readlink("/proc/self/exe", name_buf, 511)]=0;
  prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
  int child_pid = fork();
  if (!child_pid) {
    dup2(2,1); // redirect output to stderr
    fprintf(stdout,"stack trace for %s pid=%s\n",name_buf,pid_buf);
    execlp("gdb", "gdb", "--batch", "-n",
           "-ex", "thread apply all bt",
           name_buf, pid_buf, NULL);
    abort(); /* If gdb failed to start */
  } else {
    waitpid(child_pid,NULL,0);
  }
}
}
