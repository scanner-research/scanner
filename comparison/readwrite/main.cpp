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

#include "scanner/util/util.h"


#include <cstdlib>
#include <stdio.h>
#include <fstream>
#include <cassert>
#include <unistd.h>

const size_t WRITE_SIZE = 16 * 1024 * 1024;
const size_t ITERS = 1024;

char data[WRITE_SIZE];

void ofstream_test(const std::string& path) {
  std::ofstream outfile(path,
                        std::fstream::binary | std::fstream::trunc);
  assert(outfile.good());
  for (int i = 0; i < ITERS; ++i) {
    outfile.write(data, WRITE_SIZE);
  }
  outfile.flush();
  outfile.close();
}

void fopen_test(const std::string& path) {
  FILE* fp = fopen(path.c_str(), "w");
  assert(fp != nullptr);
  for (int i = 0; i < ITERS; ++i) {
    fwrite(data, 1, WRITE_SIZE, fp);
  }
  fflush(fp);
  fclose(fp);
}

void flush() {
  sync();
}

int main(int argc, char** argv) {
  auto ofstream1_start = scanner::now();
  ofstream_test("/tmp/readwrite_of1");
  flush();
  double ofstream1_time = scanner::nano_since(ofstream1_start);

  auto c1_start = scanner::now();
  fopen_test("/tmp/readwrite_c1");
  flush();
  double c1_time = scanner::nano_since(c1_start);

  auto c2_start = scanner::now();
  fopen_test("/tmp/readwrite_c2");
  flush();
  double c2_time = scanner::nano_since(c2_start);

  auto ofstream2_start = scanner::now();
  ofstream_test("/tmp/readwrite_of2");
  flush();
  double ofstream2_time = scanner::nano_since(ofstream2_start);

  size_t total_data = WRITE_SIZE * ITERS;
  printf("ofstream1: %.3f\n",
         (total_data / (1024.0 * 1024)) / (ofstream1_time / 1000000000));
  printf("ofstream2: %.3f\n",
         (total_data / (1024.0 * 1024)) / (ofstream2_time / 1000000000));
  printf("fopen1: %.3f\n",
         (total_data / (1024.0 * 1024)) / (c1_time / 1000000000));
  printf("fopen2: %.3f\n",
         (total_data / (1024.0 * 1024)) / (c2_time / 1000000000));
}
