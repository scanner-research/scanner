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

#include "lightscan/storage/storage_config.h"
#include "lightscan/storage/storage_backend.h"

#include "googleapis/client/transport/http_transport.h"
#include "googleapis/client/auth/oauth2_service_authorization.h"
#include "googleapis/client/auth/oauth2_authorization.h"
#include "google/storage_api/storage_service.h"

#include <string>

namespace lightscan {

///////////////////////////////////////////////////////////////////////////////
/// GCSConfig
struct GCSConfig : public StorageConfig {
  std::string certificates_path;
  std::string key;
  std::string bucket;
};

////////////////////////////////////////////////////////////////////////////////
/// GCSStorage
class GCSStorage : public StorageBackend {
public:
  GCSStorage(GCSConfig config);

  ~GCSStorage();

  /* make_random_read_file
   *
   */
  StoreResult make_random_read_file(
    const std::string& name,
    RandomReadFile*& file) override;

  /* make_write_file
   *
   */
  StoreResult make_write_file(
    const std::string& name,
    WriteFile*& file) override;

protected:
  const std::string certificates_path_;
  const std::string key_;
  const std::string bucket_;

private:
  std::unique_ptr<googleapis::client::HttpTransportLayerConfig> config_;
  std::unique_ptr<googleapis::client::HttpTransport> transport_;
  std::unique_ptr<googleapis::client::OAuth2ServiceAccountFlow> flow_;
  std::unique_ptr<google_storage_api::StorageService> service_;
  googleapis::client::OAuth2Credential credential_;
};

}
