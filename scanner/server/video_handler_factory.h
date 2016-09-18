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

#pragma once

#include "scanner/server/video_handler.h"
#include "scanner/server/video_handler_stats.h"
#include "scanner/util/common.h"

#include <folly/Memory.h>
#include <folly/Portability.h>
#include <folly/io/async/EventBaseManager.h>
#include <proxygen/httpserver/HTTPServer.h>
#include <proxygen/httpserver/RequestHandlerFactory.h>
#include <unistd.h>

namespace scanner {

class VideoHandlerFactory : public proxygen::RequestHandlerFactory {
 public:
  VideoHandlerFactory(
    storehouse::StorageConfig* storage_config,
    const std::string& job_name);

  void onServerStart(folly::EventBase* evb) noexcept override;

  void onServerStop() noexcept override;

  proxygen::RequestHandler* onRequest(
    proxygen::RequestHandler*,
    proxygen::HTTPMessage*) noexcept override;

 private:
  storehouse::StorageConfig* storage_config_;
  std::string job_name_;
  folly::ThreadLocalPtr<VideoHandlerStats> stats_;
};

}
