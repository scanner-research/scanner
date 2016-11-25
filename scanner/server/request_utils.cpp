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

#include "scanner/server/request_utils.h"

#include <folly/Memory.h>

#include <boost/regex.hpp>
#include <fstream>
#include <iostream>

namespace pg = proxygen;

namespace scanner {

void serve_static(const std::string& static_root, const std::string& path,
                  pg::HTTPMessage* message, pg::ResponseBuilder& response) {
  const std::map<std::string, std::string> mime_types = {
      {"html", "text/html"},
      {"js", "application/x-javascript"},
      {"css", "text/css"},
  };

  std::string path_ext = "";
  {
    static boost::regex ext_regex(".*[.](.*)$");

    boost::smatch match_result;
    if (boost::regex_match(path, match_result, ext_regex)) {
      if (match_result[1].length() != 0) {
        path_ext = match_result[1].str();
      }
    }
  }
  std::string mime_type;
  {
    auto it = mime_types.find(path_ext);
    if (it != mime_types.end()) {
      mime_type = mime_types.at(path_ext);
    } else {
      mime_type = "application/octet-stream";
    }
  }

  std::ifstream file{static_root + path, std::fstream::binary};

  if (!file) {
    std::cerr << "Could not open " << path << " for serving!" << std::endl;
    response.status(404, "Not Found");
    return;
  }

  std::streampos file_size = file.tellg();
  file.seekg(0, std::ios::end);
  file_size = file.tellg() - file_size;

  file.seekg(0);

  std::unique_ptr<folly::IOBuf> buffer{folly::IOBuf::createCombined(file_size)};
  buffer->append(file_size);
  file.read(reinterpret_cast<char*>(buffer->writableData()), file_size);

  response.status(200, "OK")
      .header(pg::HTTP_HEADER_CONTENT_TYPE, mime_type)
      .body(std::move(buffer));
}
}
