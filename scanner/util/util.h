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

#include <libgen.h>
#include <stdio.h>
#include <sys/stat.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <string>
#include <sstream>
#include <vector>

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
/// Path utils
inline std::string dirname_s(const std::string& path) {
  char* path_copy = strdup(path.c_str());
  char* dir = dirname(path_copy);
  return std::string(dir);
}

inline std::string basename_s(const std::string& path) {
  char* path_copy = strdup(path.c_str());
  char* base = basename(path_copy);
  return std::string(base);
}

int mkdir_p(const char* path, mode_t mode);

void temp_file(FILE** file, std::string& name);

///////////////////////////////////////////////////////////////////////////////
/// String processing
inline void split(const std::string &s, char delim,
                  std::vector<std::string> &elems) {
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
}

inline std::vector<std::string> split(const std::string &s, char delim) {
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

///////////////////////////////////////////////////////////////////////////////
/// MPI utils
inline bool is_master(int rank) { return rank == 0; }

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
}
