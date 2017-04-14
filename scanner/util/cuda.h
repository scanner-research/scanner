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

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#include <glog/logging.h>

#ifdef HAVE_CUDA
#define CUDA_PROTECT(s) (s);
#else
#define CUDA_PROTECT(s) \
  { LOG(FATAL) << "Cuda not enabled."; }
#endif

#ifdef HAVE_CUDA

#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sys/prctl.h>

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

#define CU_CHECK(ans) \
  { cuAssert((ans), __FILE__, __LINE__); }

inline void cuAssert(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) {
    print_trace();
    LOG(FATAL) << "GPUassert: "
               << cudaGetErrorString(code) << " "
               << file << " "
               << line;
  }
}

#define CUD_CHECK(ans) \
  { cudAssert((ans), __FILE__, __LINE__); }

inline void cudAssert(CUresult code, const char* file, int line) {
  if (code != CUDA_SUCCESS) {
    print_trace();
    const char *err_str;
    cuGetErrorString(code, &err_str);
    LOG(FATAL) << "GPUassert: " << err_str << " " << file << " " << line;
  }
}

#endif
