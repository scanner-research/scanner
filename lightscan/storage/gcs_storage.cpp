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

#include "vale/storage/gcs_storage.h"
#include "vale/util/util.h"

#include "googleapis/client/auth/file_credential_store.h"
#include "googleapis/client/auth/oauth2_authorization.h"
#include "googleapis/client/auth/oauth2_service_authorization.h"
#include "googleapis/client/data/data_reader.h"
#if HAVE_OPENSSL
#include "googleapis/client/data/openssl_codec.h"
#endif
#include "googleapis/client/transport/curl_http_transport.h"
#include "googleapis/client/transport/http_authorization.h"
#include "googleapis/client/transport/http_transport.h"
#include "googleapis/client/transport/http_request_batch.h"
#include "googleapis/client/util/status.h"
#include "googleapis/client/util/uri_utils.h"
#include "googleapis/strings/strcat.h"
#include "googleapis/util/status.h"

#include <cassert>

using googleapis::client::HttpTransport;
using googleapis::client::HttpTransportLayerConfig;
using googleapis::client::HttpTransportFactory;
using googleapis::client::HttpRequest;
using googleapis::client::HttpResponse;
using googleapis::client::OAuth2RequestOptions;
using googleapis::client::OAuth2ServiceAccountFlow;
using googleapis::client::CurlHttpTransportFactory;
using googleapis::client::CurlHttpTransportFactory;
using googleapis::client::OAuth2Credential;

using google_storage_api::ObjectsResource_InsertMethod;
using google_storage_api::ObjectsResource_GetMethod;

namespace vale {
namespace internal {

////////////////////////////////////////////////////////////////////////////////
/// GCSRandomReadFile
class GCSRandomReadFile : public RandomReadFile {
public:
  GCSRandomReadFile(
    HttpTransport* transport,
    OAuth2Credential* credential,
    const std::string& object_name,
    const std::string& media_url)
    : transport_(transport),
      credential_(credential),
      object_name_(object_name),
      media_url_(media_url)
  {

  }

  StoreResult read(
    uint64_t offset,
    size_t size,
    char* data,
    size_t& size_read) override
  {
    std::unique_ptr<HttpRequest> request(
      transport_->NewHttpRequest(HttpRequest::GET));
    request->set_url(media_url_.c_str());
    request->set_credential(credential_);
    // Set range (inclusive)
    std::string range_str =
      "bytes=" + std::to_string(offset) + "-" +
      std::to_string(offset + size - 1);
    request->AddHeader("Range", range_str);

    googleapis::util::Status status = request->Execute();
    HttpResponse *response = request->response();
    if (!status.ok()) {
      switch (status.error_code()) {
      case googleapis::util::error::DEADLINE_EXCEEDED: {
        // Timeout error
        return StoreResult::TransientFailure;
      } default: {
      }
      }

      // 416 Requested Range Not Satisfiable
      if (response->http_code() == 416) {
        size_read = 0;
        return StoreResult::EndOfFile;
      }
      log_vale.fatal("GCSRandomRead (offset %lu, size %lu) error: %s\n",
                     offset, size, status.error_message().c_str());
      assert(status.ok());
    }

    std::string body;
    response->GetBodyString(&body);
    size_read = body.size();

    memcpy(data, body.data(), body.size());

    if (size_read < size) {
      return StoreResult::EndOfFile;
    } else {
      return StoreResult::Success;
    }
  }

private:
  HttpTransport* transport_;
  OAuth2Credential* credential_;
  const std::string object_name_;
  const std::string media_url_;
};

////////////////////////////////////////////////////////////////////////////////
/// GCSAppendWriteFile
class GCSWriteFile : public WriteFile {
public:
  GCSWriteFile(
    google_storage_api::StorageService* service,
    OAuth2Credential* credential,
    const std::string& bucket_name,
    const std::string& object_name)
    : service_(service),
      credential_(credential),
      bucket_name_(bucket_name),
      object_name_(object_name)
  {
  }

  StoreResult append(size_t size, const char* data) override {
    data_buffer_.insert(data_buffer_.end(), data, data + size);

    return StoreResult::Success;
  }

  StoreResult save() override {
    // Make media reader
    googleapis::client::DataReader* reader =
      googleapis::client::NewManagedInMemoryDataReader(
        googleapis::StringPiece(data_buffer_.data(), data_buffer_.size()),
        nullptr);

    // Create request to insert object
    std::unique_ptr<ObjectsResource_InsertMethod> method(
      service_->get_objects().NewInsertMethod(
        credential_,
        googleapis::StringPiece(bucket_name_),
        nullptr,
        nullptr,
        reader));

    method->set_name(object_name_);

    google_storage_api::Object* response_obj =
      google_storage_api::Object::New();
    googleapis::util::Status status =
      method->ExecuteAndParseResponse(response_obj);
    delete response_obj;
    if (!status.ok()) {
      switch (status.error_code()) {
      case googleapis::util::error::DEADLINE_EXCEEDED: {
        // Timeout error
        return StoreResult::TransientFailure;
      } default: {
      }
      }

      log_vale.fatal("GCSWriteFile: save failed for object %s: %s",
                      object_name_.c_str(), status.error_message().c_str());
      assert(status.ok());
    }
    return StoreResult::Success;
  }

private:
  google_storage_api::StorageService* service_;
  OAuth2Credential* credential_;

  const std::string bucket_name_;
  const std::string object_name_;
  std::vector<char> data_buffer_;
};

////////////////////////////////////////////////////////////////////////////////
/// GCSStorage
GCSStorage::GCSStorage(
  GCSConfig config)
  : certificates_path_(config.certificates_path),
    key_(config.key),
    bucket_(config.bucket)
{
  googleapis::util::Status status;

  HttpTransportLayerConfig* transport_config = new HttpTransportLayerConfig;
  transport_config->mutable_default_transport_options()->set_cacerts_path(
    certificates_path_);

  config_.reset(transport_config);
  HttpTransportFactory* factory =
    new CurlHttpTransportFactory(config_.get());
  config_->ResetDefaultTransportFactory(factory);

  transport_.reset(config_->NewDefaultTransportOrDie());

  flow_.reset(new OAuth2ServiceAccountFlow(
                config_->NewDefaultTransportOrDie()));
  flow_->InitFromJson(key_);
  flow_->set_default_scopes(
    google_storage_api::StorageService::SCOPES::DEVSTORAGE_FULL_CONTROL);

  service_.reset(new google_storage_api::StorageService(
                   config_->NewDefaultTransportOrDie()));

  OAuth2RequestOptions options;
  status = flow_->PerformRefreshToken(options, &credential_);

  if (!status.ok()) {
    std::cerr << "Refresh tokens failed: " << status.error_message();
    assert(status.ok());
  }

  credential_.set_flow(flow_.get());
}

GCSStorage::~GCSStorage() {
}

StoreResult GCSStorage::make_random_read_file(
  const std::string& name,
  RandomReadFile*& file)
{
  // Create request to get object metadata
  std::unique_ptr<ObjectsResource_GetMethod> method(
    service_->get_objects().NewGetMethod(&credential_,
                                         googleapis::StringPiece(bucket_),
                                         googleapis::StringPiece(name)));

  google_storage_api::Object* response_obj =
    google_storage_api::Object::New();
  googleapis::util::Status status =
    method->ExecuteAndParseResponse(response_obj);
  if (!status.ok()) {
    switch (status.error_code()) {
    case googleapis::util::error::DEADLINE_EXCEEDED: {
      // Timeout error
      return StoreResult::TransientFailure;
    } default: {
    }
    }
    log_vale.fatal("GCSStorage: FATAL: make_random_read_file (%s) error: %s\n",
                   name.c_str(), status.error_message().c_str());
    assert(status.ok());
  }

  std::string media_url = response_obj->get_media_link().ToString();
  assert(!media_url.empty());

  file = new GCSRandomReadFile(transport_.get(), &credential_, name, media_url);

  return StoreResult::Success;
}

StoreResult GCSStorage::make_write_file(
  const std::string& name,
  WriteFile*& file)
{
  file = new GCSWriteFile(service_.get(), &credential_, bucket_, name);
  return StoreResult::Success;
}

}
}
