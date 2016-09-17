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

#include "scanner/server/video_handler.h"
#include "scanner/server/video_handler_stats.h"
#include "scanner/server/request_utils.h"
#include "scanner/server/results_parser.h"
#include "scanner/util/storehouse.h"
#include "scanner/util/common.h"

#include "storehouse/storage_backend.h"

#include "scanner/parsers/imagenet_parser.h"

#include <proxygen/httpserver/RequestHandler.h>
#include <proxygen/httpserver/ResponseBuilder.h>
#include <proxygen/lib/http/HTTPCommonHeaders.h>

#include <folly/dynamic.h>
#include <folly/json.h>

#include <boost/regex.hpp>

#include <map>
#include <fstream>
#include <unistd.h>

namespace pg = proxygen;

using storehouse::StorageConfig;
using storehouse::StorageBackend;
using storehouse::RandomReadFile;
using storehouse::WriteFile;
using storehouse::StoreResult;

namespace scanner {

namespace {

DatasetDescriptor read_dataset_descriptor(
  StorageBackend* storage,
  const std::string& dataset_name)
{
  const std::string dataset_path =
    dataset_descriptor_path(dataset_name);
  std::unique_ptr<RandomReadFile> file;
  make_unique_random_read_file(storage, dataset_path, file);

  u64 pos = 0;
  return deserialize_dataset_descriptor(file.get(), pos);
}

DatasetItemMetadata read_dataset_item_metadata(
  StorageBackend* storage,
  const std::string& dataset_name,
  const std::string& item_name)
{
  const std::string metadata_path =
    dataset_item_metadata_path(dataset_name, item_name);
  std::unique_ptr<RandomReadFile> file;
  make_unique_random_read_file(storage, metadata_path, file);

  u64 pos = 0;
  return deserialize_dataset_item_metadata(file.get(), pos);
}

DatasetItemWebTimestamps read_dataset_item_web_timestamps(
  StorageBackend* storage,
  const std::string& dataset_name,
  const std::string& item_name)
{
  const std::string metadata_path =
    dataset_item_video_timestamps_path(dataset_name, item_name);
  std::unique_ptr<RandomReadFile> file;
  make_unique_random_read_file(storage, metadata_path, file);

  u64 pos = 0;
  return deserialize_dataset_item_web_timestamps(file.get(), pos);
}

JobDescriptor read_job_descriptor(
  StorageBackend* storage,
  const std::string& job_name)
{
  const std::string job_path = job_descriptor_path(job_name);
  std::unique_ptr<RandomReadFile> file;
  make_unique_random_read_file(storage, job_path, file);

  u64 pos = 0;
  return deserialize_job_descriptor(file.get(), pos);
}

}

VideoHandler::VideoHandler(VideoHandlerStats* stats, StorageConfig* config):
  stats_(stats),
  storage_(StorageBackend::make_from_config(config))
{
}

void VideoHandler::onRequest(std::unique_ptr<pg::HTTPMessage> message)
  noexcept
{
  message_ = std::move(message);
  std::cout << "on request " << message_->getPath() << std::endl;
  fflush(stdout);
  stats_->recordRequest();
}

void VideoHandler::onBody(std::unique_ptr<folly::IOBuf> body) noexcept {
  if (body_) {
    body_->prependChain(std::move(body));
  } else {
    body_ = std::move(body);
  }
}

void VideoHandler::onEOM() noexcept {
  const pg::HTTPHeaders& headers = message_->getHeaders();

  std::string path = message_->getPath();
  if (path == "/") {
    path = "/index.html";
  }

  // Static job name and parser for now
  const i32 job_id = 1;
  const std::string job_name = "test_small";

  ResultsParser* parser = new ImagenetParser();

  static const boost::regex jobs_regex("^/jobs");
  static const boost::regex videos_regex("^/videos");
  static const boost::regex features_regex(
    "^/jobs/([[:digit:]]+)/features/([[:digit:]]+)");
  static const boost::regex media_regex(
    "^/jobs/([[:digit:]]+)/media/([[:digit:]]+).mp4");

  auto response = pg::ResponseBuilder(downstream_);

  boost::smatch match_result;
  if (boost::regex_match(path, match_result, jobs_regex)) {
    JobDescriptor job_descriptor =
      read_job_descriptor(storage_.get(), job_name);

    folly::dynamic json = folly::dynamic::object;

    bool bad = false;
    static const boost::regex ex("^/jobs/([[:digit:]]+)");
    if (boost::regex_match(path, match_result, ex)) {
      // Requesting a specific video's metadata
      i32 item_id = std::atoi(match_result[1].str().c_str());

      if (item_id != job_id) {
        response.status(400, "Bad Request");
        bad = true;
      } else {
        // List all jobs, which right now is only one hardcoded one
        folly::dynamic meta = folly::dynamic::object;

        meta["id"] = job_id;
        meta["name"] = job_name;
        meta["dataset"] = job_descriptor.dataset_name;
        //meta["featureType"] = "detection";
        meta["featureType"] = "classification";

        json = meta;
      }
    } else {
      // List all jobs, which right now is only one hardcoded one
      json = folly::dynamic::array();

      folly::dynamic meta = folly::dynamic::object;

      meta["id"] = job_id;
      meta["name"] = job_name;
      meta["dataset"] = job_descriptor.dataset_name;
      //meta["featureType"] = "detection";
      meta["featureType"] = "classification";

      json.push_back(meta);
    }

    if (!bad) {
      std::string body = folly::toJson(json);

      response
        .status(200, "OK")
        .body(body);
    }
  } else if (boost::regex_match(path, match_result, videos_regex)) {
    if (!message_->hasQueryParam("job_id")) {
      response.status(400, "Bad Request");
    } else {
      i32 requested_job_id = message_->getIntQueryParam("job_id");
      if (requested_job_id != job_id) {
        response.status(400, "Bad Request");
      } else {
        JobDescriptor job_descriptor =
          read_job_descriptor(storage_.get(), job_name);
        DatasetDescriptor dataset_descriptor =
          read_dataset_descriptor(storage_.get(), job_descriptor.dataset_name);

        auto item_to_json = [](i32 item_id,
                               const std::string& item_name,
                               const std::string& media_path,
                               const DatasetItemMetadata& item)
          -> folly::dynamic
          {
            folly::dynamic meta = folly::dynamic::object;
            meta["id"] = item_id;
            meta["name"] = item_name;
            meta["mediaPath"] = media_path;
            meta["frames"] = item.frames;
            meta["width"] = item.width;
            meta["height"] = item.height;
            return meta;
          };

        folly::dynamic json = folly::dynamic::object;

        bool bad = false;
        static const boost::regex ex("^/videos/([[:digit:]]+)");
        if (boost::regex_match(path, match_result, ex)) {
          // Requesting a specific video's metadata
          i32 item_id = std::atoi(match_result[1].str().c_str());

          if (item_id >= dataset_descriptor.item_names.size()) {
            response.status(400, "Bad Request");
            bad = true;
          } else {
            const std::string& item_name =
              dataset_descriptor.item_names[item_id];
            const std::string& media_path =
              "jobs/" + std::to_string(requested_job_id) +
              "/media/" + item_name + ".mp4";

            DatasetItemMetadata item =
              read_dataset_item_metadata(
                storage_.get(),
                job_descriptor.dataset_name,
                item_name);

            json = item_to_json(item_id, item_name, media_path, item);
          }
        } else {
          // Requesting all videos metadata
          json = folly::dynamic::array();
          i32 item_id = 0;
          for (const std::string& item_name : dataset_descriptor.item_names) {
            const std::string& media_path =
              "jobs/" + std::to_string(requested_job_id) +
              "/media/" + item_name + ".mp4";

            DatasetItemMetadata item =
              read_dataset_item_metadata(
                storage_.get(),
                job_descriptor.dataset_name,
                item_name);

            folly::dynamic meta =
              item_to_json(item_id++, item_name, media_path, item);

            json.push_back(meta);
          }
        }
        if (!bad) {
          std::string body = folly::toJson(json);

          response
            .status(200, "OK")
            .body(body);
        }
      }
    }
  } else if (boost::regex_match(path, match_result, features_regex)) {
    if (!(message_->hasQueryParam("start") &&
          message_->hasQueryParam("end")))
    {
      response.status(400, "Bad Request");
    } else {
      i32 requested_job_id = std::atoi(match_result[1].str().c_str());
      if (requested_job_id != job_id) {
        response.status(400, "Bad Request");
      } else {
        JobDescriptor job_descriptor =
          read_job_descriptor(storage_.get(), job_name);
        DatasetDescriptor dataset_descriptor =
          read_dataset_descriptor(storage_.get(), job_descriptor.dataset_name);

        i32 start_frame = message_->getIntQueryParam("start");
        i32 end_frame = message_->getIntQueryParam("end");

        i32 stride = 1;
        if (message_->hasQueryParam("stride")) {
          stride = message_->getIntQueryParam("stride");
        }

        i32 filtered_category = -1;
        if (message_->hasQueryParam("category")) {
          filtered_category = message_->getIntQueryParam("category");
        }

        f32 threshold = -1;
        if (message_->hasQueryParam("threshold")) {
          threshold = folly::to<f32>(message_->getQueryParam("threshold"));
        }

        // if (message_->has("sampling_filter")) {
        // }

        i32 item_id = std::atoi(match_result[2].str().c_str());
        std::string item_name = dataset_descriptor.item_names[item_id];
        const std::vector<std::tuple<i32, i32>>& intervals =
          job_descriptor.intervals[item_name];
        std::vector<std::string> output_names = parser->get_output_names();

        // Get the mapping from frames to timestamps to guide UI toward
        // exact video frame corresponding to the returned features. Needed
        // because some UIs (looking at you html5) do not support frame
        // accurate seeking.
        DatasetItemWebTimestamps timestamps =
          read_dataset_item_web_timestamps(storage_.get(),
                                           job_descriptor.dataset_name,
                                           item_name);


        folly::dynamic feature_classes = folly::dynamic::array();

        std::vector<std::vector<u8>> output_buffers(
          output_names.size());
        std::vector<size_t> output_buffers_pos(output_names.size());
        i64 total_frames_in_output_buffer = 0;
        i64 current_frame_in_output_buffer = 0;

        i32 current_frame = start_frame;
        while (current_frame < end_frame) {
          if (current_frame_in_output_buffer >= total_frames_in_output_buffer) {
            // Find the correct interval
            size_t i;
            for (i = 0; i < intervals.size(); ++i) {
              if (current_frame >= std::get<0>(intervals[i]) &&
                  current_frame < std::get<1>(intervals[i]))
              {
                break;
              }
            }
            const auto& interval = intervals[i];

            i64 start = std::get<0>(interval);
            i64 end = std::get<1>(interval);

            // Skip past frames at the beginning to fulfill our striding
            current_frame_in_output_buffer =
              current_frame_in_output_buffer - total_frames_in_output_buffer;
            total_frames_in_output_buffer = end - start;

            for (size_t output_index = 0;
                 output_index < output_names.size();
                 ++output_index)
            {
              std::string output_path = job_item_output_path(
                job_name,
                item_name,
                output_names[output_index],
                start,
                end);

              std::unique_ptr<RandomReadFile> file;
              make_unique_random_read_file(storage_.get(), output_path, file);

              u64 pos = 0;
              output_buffers[output_index] = read_entire_file(file.get(), pos);
              output_buffers_pos[output_index] = 0;
            }
          }

          // Extract row of feature vectors from output buffers
          std::vector<u8*> output(output_names.size());
          std::vector<i64> output_sizes(output_names.size());
          for (size_t output_idx = 0;
               output_idx < output_names.size();
               ++output_idx)
          {
            size_t offset = output_buffers_pos[output_idx];
            output_sizes[output_idx] =
              *((i64*)(output_buffers[output_idx].data() + offset));
            output[output_idx] =
              output_buffers[output_idx].data() + offset + sizeof(i64);

            output_buffers_pos[output_idx] +=
              output_sizes[output_idx] + sizeof(i64);
          }

          // Process vectors
          folly::dynamic feature_data = folly::dynamic::object();
          feature_data["frame"] = current_frame;
          feature_data["time"] =
            timestamps.pts_timestamps[current_frame] /
            static_cast<f64>(timestamps.time_base_denominator);
          feature_data["data"] = folly::dynamic::object();

          parser->parse_output(output, output_sizes, feature_data["data"]);

          feature_classes.push_back(feature_data);

          current_frame += stride;
          current_frame_in_output_buffer += stride;
        }

        std::cout << "to json " << std::endl;
        std::string body = folly::toJson(feature_classes);
        std::cout << "finished to json " << std::endl;

        response
          .status(200, "OK")
          .body(body);
      }
    }
  } else if (boost::regex_match(path, match_result, media_regex)) {
    i32 requested_job_id = std::atoi(match_result[1].str().c_str());
    if (requested_job_id != job_id) {
      response.status(400, "Bad Request");
    } else {
      JobDescriptor job_descriptor =
        read_job_descriptor(storage_.get(), job_name);
      DatasetDescriptor dataset_descriptor =
        read_dataset_descriptor(storage_.get(), job_descriptor.dataset_name);

      std::string media_path = match_result[2].str();

      const std::string mp4_path =
        dataset_item_video_path(job_descriptor.dataset_name, media_path);
      std::unique_ptr<RandomReadFile> file;
      make_unique_random_read_file(storage_.get(), mp4_path, file);

      // movie.mp4
      std::string mime_type = "video/mp4";

      StoreResult result;

      u64 file_size;
      EXP_BACKOFF(file->get_size(file_size), result);
      exit_on_error(result);

      i64 byte_range_start = -1;
      i64 byte_range_end = -1;
      if (headers.exists(pg::HTTP_HEADER_RANGE)) {
        static boost::regex range_regex("^bytes=([0-9]*)-([0-9]*)");

        const std::string& range_str =
          headers.getSingleOrEmpty(pg::HTTP_HEADER_RANGE);

        boost::smatch match_result;
        if (boost::regex_match(range_str, match_result, range_regex)) {
          if (match_result[1].length() != 0) {
            byte_range_start = std::atoi(match_result[1].str().c_str());
          }
          if (match_result[2].length() != 0) {
            byte_range_end = std::atoi(match_result[2].str().c_str());
          }
        }
        if (byte_range_start == -1) {
          // Last N bytes
          byte_range_start = file_size - byte_range_end;
        } else if (byte_range_end == -1) {
          // Until end
          byte_range_end = static_cast<i64>(file_size) - 1;
        }
      } else {
        byte_range_start = 0;
        byte_range_end = static_cast<i64>(file_size) - 1;
      }

      u64 read_size = byte_range_end - byte_range_start + 1;
      std::unique_ptr<folly::IOBuf> buffer{
        folly::IOBuf::createCombined(read_size)};
      buffer->append(read_size);
      size_t size_read;

      std::cout << "reading request " << message_->getPath() << std::endl;
      fflush(stdout);
      u64 pos = byte_range_start;
      read(file.get(), buffer->writableData(), read_size, pos);
      std::cout << "finished reading request " << message_->getPath() << std::endl;
      fflush(stdout);

      response
        .status(206, "Partial Content")
        .header(pg::HTTP_HEADER_CONTENT_TYPE,
                mime_type)
        .header(pg::HTTP_HEADER_ACCEPT_RANGES, "bytes")
        .header(pg::HTTP_HEADER_CONTENT_RANGE,
                "bytes " +
                std::to_string(byte_range_start) + "-" +
                std::to_string(byte_range_end) + "/" +
                std::to_string(file_size))
        .body(std::move(buffer));
    }
  } else {
    // Serve static files
    serve_static("www", path, message_.get(), response);
  }

  response
    .header("Request-Number",
            folly::to<std::string>(stats_->getRequestCount()))
    .sendWithEOM();
}

void VideoHandler::onUpgrade(pg::UpgradeProtocol protocol) noexcept {
  // handler doesn't support upgrades
}

void VideoHandler::requestComplete() noexcept {
  std::cout << "finished request " << message_->getPath() << std::endl;
  fflush(stdout);
  delete this;
}

void VideoHandler::onError(pg::ProxygenError err) noexcept {
  std::cout << "on error request " << message_->getPath() << std::endl;
  delete this;
}

}
