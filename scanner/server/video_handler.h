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

#include "scanner/util/common.h"
#include "scanner/util/db.h"
#include "storehouse/storage_backend.h"

#include <folly/Memory.h>
#include <proxygen/httpserver/RequestHandler.h>
#include <proxygen/httpserver/ResponseBuilder.h>

#include <boost/regex.hpp>

namespace proxygen {
class ResponseHandler;
}

namespace scanner {

class VideoHandlerStats;

class VideoHandler : public proxygen::RequestHandler {
 public:
  explicit VideoHandler(VideoHandlerStats* stats,
                        storehouse::StorageConfig* config);

  void onRequest(
      std::unique_ptr<proxygen::HTTPMessage> message) noexcept override;

  void onBody(std::unique_ptr<folly::IOBuf> body) noexcept override;

  void onEOM() noexcept override;

  void onUpgrade(proxygen::UpgradeProtocol proto) noexcept override;

  void requestComplete() noexcept override;

  void onError(proxygen::ProxygenError err) noexcept override;

 private:
  void handle_datasets(const DatabaseMetadata& meta, const std::string& path,
                       proxygen::ResponseBuilder& response);

  void handle_jobs(const DatabaseMetadata& meta, i32 dataset_id,
                   const std::string& path,
                   proxygen::ResponseBuilder& response);

  void handle_features(const DatabaseMetadata& meta, i32 dataset_id, i32 job_id,
                       i32 video_id, const std::string& path,
                       proxygen::ResponseBuilder& response);

  void handle_videos(const DatabaseMetadata& meta, i32 dataset_id,
                     const std::string& path,
                     proxygen::ResponseBuilder& response);

  void handle_media(const DatabaseMetadata& meta, i32 dataset_id,
                    const std::string& media_path, const std::string& path,
                    proxygen::ResponseBuilder& response);

  const boost::regex id_regex{"^/([[:digit:]]+)"};

  VideoHandlerStats* const stats_{nullptr};
  std::unique_ptr<storehouse::StorageBackend> storage_{nullptr};

  std::unique_ptr<proxygen::HTTPMessage> message_;
  std::unique_ptr<folly::IOBuf> body_;
};
}
