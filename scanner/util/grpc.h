/* Licensed under the Apache License, Version 2.0 (the "License");
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

#include <cstdlib>
#include <unistd.h>

namespace scanner {

#define GRPC_BACKOFF(expression__, status__)        \
  GRPC_BACKOFF_TIMEOUT(expression__, status__, 64, 0)

#define GRPC_BACKOFF_D(expression__, status__, deadline__)       \
  GRPC_BACKOFF_TIMEOUT(expression__, status__, 64, deadline__)

#define GRPC_BACKOFF_TIMEOUT(expression__, status__, timeout__, deadline__) \
  do {                                                                      \
    int sleep_debt__ = 1;                                                   \
    while (true) {                                                          \
      grpc::ClientContext ctx;                                              \
      if (deadline__ > 0) {                                                 \
        std::chrono::system_clock::time_point deadline =                    \
            std::chrono::system_clock::now() +                              \
            std::chrono::seconds(deadline__);                               \
        ctx.set_deadline(deadline);                                         \
      }                                                                     \
      const grpc::Status result__ = (expression__);                         \
      if (result__.error_code() == grpc::StatusCode::UNAVAILABLE) {         \
        double sleep_time__ =                                               \
            (sleep_debt__ + (static_cast<double>(rand()) / RAND_MAX));      \
        if (sleep_debt__ < (timeout__)) {                                   \
          sleep_debt__ *= 2;                                                \
        } else {                                                            \
          LOG(WARNING) << "GRPC_BACKOFF: reached max backoff.";             \
          status__ = result__;                                              \
          break;                                                            \
        }                                                                   \
        LOG(WARNING) << "GRPC_BACKOFF: transient failure, sleeping for "    \
                     << sleep_time__ << " seconds.";                        \
        usleep(sleep_time__ * 1000000);                                     \
        continue;                                                           \
      }                                                                     \
      status__ = result__;                                                  \
      break;                                                                \
    }                                                                       \
  } while (0);
}
