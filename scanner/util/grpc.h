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

namespace scanner {

#define GRPC_BACKOFF(expression__, status__)                             \
  do {                                                                   \
    int sleep_debt__ = 1;                                                \
    while (true) {                                                       \
      grpc::ClientContext ctx;                                           \
      const grpc::Status result__ = (expression__);                      \
      if (result__.error_code() == grpc::StatusCode::UNAVAILABLE) {      \
        double sleep_time__ =                                            \
            (sleep_debt__ + (static_cast<double>(rand()) / RAND_MAX));   \
        if (sleep_debt__ < 64) {                                         \
          sleep_debt__ *= 2;                                             \
        } else {                                                         \
          LOG(FATAL) << "GRPC_BACKOFF: reached max backoff.";            \
          exit(1);                                                       \
        }                                                                \
        LOG(WARNING) << "GRPC_BACKOFF: transient failure, sleeping for " \
                     << sleep_time__ << " seconds.";                     \
        usleep(sleep_time__ * 1000000);                                  \
        continue;                                                        \
      }                                                                  \
      status__ = result__;                                               \
      break;                                                             \
    }                                                                    \
  } while (0);
}
