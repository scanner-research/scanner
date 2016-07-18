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
#include <chrono>
#include <string>
#include <sys/stat.h>
#include <stdio.h>
#include <cstring>
#include <libgen.h>

namespace lightscan {

class Logger {
public:
  void spew(const char *fmt, ...) __attribute__((format (printf, 2, 3)));
  void debug(const char *fmt, ...) __attribute__((format (printf, 2, 3)));
  void info(const char *fmt, ...) __attribute__((format (printf, 2, 3)));
  void print(const char *fmt, ...) __attribute__((format (printf, 2, 3)));
  void warning(const char *fmt, ...) __attribute__((format (printf, 2, 3)));
  void error(const char *fmt, ...) __attribute__((format (printf, 2, 3)));
  void fatal(const char *fmt, ...) __attribute__((format (printf, 2, 3)));

};

extern Logger log_ls;

class SpinLock {
    std::atomic_flag locked = ATOMIC_FLAG_INIT ;
public:
    void lock() {
        while (locked.test_and_set(std::memory_order_acquire)) { ; }
    }
    void unlock() {
        locked.clear(std::memory_order_release);
    }
};

inline std::chrono::time_point<std::chrono::high_resolution_clock> now() {
  return std::chrono::high_resolution_clock::now();
}

inline double nano_since(
  std::chrono::time_point<std::chrono::high_resolution_clock> then)
{
  return
    std::chrono::duration_cast<std::chrono::nanoseconds>(now() - then).count();
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

int mkdir_p(const char *path, mode_t mode);

void temp_file(FILE** file, std::string& name);

///////////////////////////////////////////////////////////////////////////////
/// pthread utils
#define THREAD_RETURN_SUCCESS() \
  do {                                           \
    void* val = malloc(sizeof(int));             \
    *((int*)val) = EXIT_SUCCESS;                 \
    pthread_exit(val);                           \
  } while (0);

///////////////////////////////////////////////////////////////////////////////
/// MPI utils
inline bool is_master(int rank) {
  return rank == 0;
}

}
