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

#include "scanner/server/video_handler_factory.h"
#include "scanner/server/video_handler_stats.h"

#include <folly/Memory.h>
#include <folly/Portability.h>
#include <folly/io/async/EventBaseManager.h>
#include <proxygen/httpserver/HTTPServer.h>
#include <proxygen/httpserver/RequestHandlerFactory.h>
#include <unistd.h>

using storehouse::StorageConfig;
using storehouse::StoreResult;

namespace scanner {

VideoHandlerFactory::VideoHandlerFactory(
  storehouse::StorageConfig* storage_config,
  const std::string& job_name)
  : storage_config_(storage_config),
    job_name_(job_name)
{
}

void VideoHandlerFactory::onServerStart(
  folly::EventBase* evb) noexcept 
{
  stats_.reset(new VideoHandlerStats);
}

void VideoHandlerFactory::onServerStop() noexcept {
  stats_.reset();
}

proxygen::RequestHandler* VideoHandlerFactory::onRequest(
  proxygen::RequestHandler*,
  proxygen::HTTPMessage*) noexcept 
{
  return new VideoHandler(stats_.get(), storage_config_, job_name_);
}

}
