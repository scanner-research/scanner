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
#include "scanner/engine.h"
#include "scanner/server/request_utils.h"
#include "scanner/server/results_parser.h"
#include "scanner/server/video_handler_stats.h"
#include "scanner/util/common.h"
#include "scanner/util/storehouse.h"
#include "scanner/util/util.h"

#include "storehouse/storage_backend.h"

#include "scanner/parsers/bbox_parser.h"
#include "scanner/parsers/facenet_parser.h"

#include <proxygen/httpserver/RequestHandler.h>
#include <proxygen/httpserver/ResponseBuilder.h>
#include <proxygen/lib/http/HTTPCommonHeaders.h>

#include <folly/dynamic.h>
#include <folly/json.h>

#include <boost/regex.hpp>

#include <fstream>
#include <map>
#include <unistd.h>

namespace pg = proxygen;

using storehouse::StorageConfig;
using storehouse::StorageBackend;
using storehouse::RandomReadFile;
using storehouse::WriteFile;
using storehouse::StoreResult;

namespace scanner {

namespace {

DatabaseMetadata read_database_metadata(StorageBackend *storage) {
  const std::string db_meta_path = database_metadata_path();
  std::unique_ptr<RandomReadFile> file;
  make_unique_random_read_file(storage, db_meta_path, file);

  u64 pos = 0;
  return deserialize_database_metadata(file.get(), pos);
}

DatasetDescriptor read_dataset_descriptor(StorageBackend *storage,
                                          const std::string &dataset_name) {
  const std::string dataset_path = dataset_descriptor_path(dataset_name);
  std::unique_ptr<RandomReadFile> file;
  make_unique_random_read_file(storage, dataset_path, file);

  u64 pos = 0;
  return deserialize_dataset_descriptor(file.get(), pos);
}

VideoMetadata read_video_metadata(StorageBackend *storage,
                                  const std::string &dataset_name,
                                  const std::string &item_name) {
  std::string metadata_path =
      dataset_item_metadata_path(dataset_name, item_name);
  std::unique_ptr<RandomReadFile> file;
  make_unique_random_read_file(storage, metadata_path, file);

  u64 pos = 0;
  return deserialize_video_metadata(file.get(), pos);
}

WebTimestamps read_web_timestamps(StorageBackend *storage,
                                  const std::string &dataset_name,
                                  const std::string &item_name) {
  const std::string metadata_path =
      dataset_item_video_timestamps_path(dataset_name, item_name);
  std::unique_ptr<RandomReadFile> file;
  make_unique_random_read_file(storage, metadata_path, file);

  u64 pos = 0;
  return deserialize_web_timestamps(file.get(), pos);
}

JobDescriptor read_job_descriptor(StorageBackend *storage,
                                  const std::string &dataset_name,
                                  const std::string &job_name) {
  const std::string job_path = job_descriptor_path(dataset_name, job_name);
  std::unique_ptr<RandomReadFile> file;
  make_unique_random_read_file(storage, job_path, file);

  u64 pos = 0;
  return deserialize_job_descriptor(file.get(), pos);
}
}

VideoHandler::VideoHandler(VideoHandlerStats *stats, StorageConfig *config)
    : stats_(stats), storage_(StorageBackend::make_from_config(config)) {}

void VideoHandler::onRequest(
    std::unique_ptr<pg::HTTPMessage> message) noexcept {
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
  const pg::HTTPHeaders &headers = message_->getHeaders();

  std::string path = message_->getPath();
  if (path == "/") {
    path = "/index.html";
  }

  DatabaseMetadata meta = read_database_metadata(storage_.get());

  static const boost::regex datasets_regex("^/datasets");

  auto response = pg::ResponseBuilder(downstream_);

  boost::smatch match_result;
  if (boost::regex_search(path, match_result, datasets_regex)) {
    std::string suffix = match_result.suffix();
    handle_datasets(meta, suffix, response);
  } else {
    // Serve static files
    serve_static("www/dist", path, message_.get(), response);
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

void VideoHandler::handle_datasets(const DatabaseMetadata &meta,
                                   const std::string &path,
                                   pg::ResponseBuilder &response) {
  bool bad = false;
  boost::smatch match_result;
  if (path.empty()) {
    folly::dynamic json = folly::dynamic::array();
    for (const auto &kv : meta.dataset_names) {
      DatasetMetadata dataset_meta =
          read_dataset_descriptor(storage_.get(), kv.second);

      folly::dynamic dataset_info = folly::dynamic::object();
      dataset_info["id"] = kv.first;
      dataset_info["name"] = kv.second;
      dataset_info["total_frames"] = dataset_meta.total_frames();
      folly::dynamic video_names = folly::dynamic::array();
      for (const std::string &path : dataset_meta.original_paths()) {
        video_names.push_back(path);
      }
      dataset_info["video_names"] = video_names;

      json.push_back(dataset_info);
    }
    std::string body = folly::toJson(json);
    response.status(200, "OK").body(body);
  } else if (boost::regex_search(path, match_result, id_regex)) {
    static const boost::regex jobs_regex("^/jobs");
    static const boost::regex videos_regex("^/videos");
    static const boost::regex media_regex("^/media/([[:digit:]]+)\\.mp4");

    std::string suffix = match_result.suffix();

    i32 dataset_id = std::atoi(match_result[1].str().c_str());
    if (meta.dataset_names.count(dataset_id) < 1) {
      response.status(400, "Bad Request");
      bad = true;
    } else {
      if (suffix.empty()) {
        // TODO(apoms): handle returning individual dataset result
        LOG(FATAL) << "NOT IMPLEMENTED";
      } else if (boost::regex_search(suffix, match_result, jobs_regex)) {
        suffix = match_result.suffix();
        handle_jobs(meta, dataset_id, suffix, response);
      } else if (boost::regex_search(suffix, match_result, videos_regex)) {
        suffix = match_result.suffix();
        handle_videos(meta, dataset_id, suffix, response);
      } else if (boost::regex_search(suffix, match_result, media_regex)) {
        std::string media_path = match_result[1];
        suffix = match_result.suffix();
        handle_media(meta, dataset_id, media_path, suffix, response);
      }
    }
  } else {
    response.status(400, "Bad Request");
  }
}

void VideoHandler::handle_jobs(const DatabaseMetadata &meta, i32 dataset_id,
                               const std::string &path,
                               pg::ResponseBuilder &response) {
  static const boost::regex features_regex("^/features/([[:digit:]]+)");

  const std::string &dataset_name = meta.dataset_names.at(dataset_id);
  const std::set<i32> &job_ids = meta.dataset_job_ids.at(dataset_id);

  boost::smatch match_result;
  if (boost::regex_search(path, match_result, id_regex)) {
    std::string suffix = match_result.suffix();
    // Asking for a specific job's information
    i32 job_id = std::atoi(match_result[1].str().c_str());
    if (job_ids.count(job_id) < 1) {
      response.status(400, "Bad Request");
      return;
    }
    if (suffix.empty()) {
      const std::string &job_name = meta.job_names.at(job_id);
      JobDescriptor job_descriptor =
          read_job_descriptor(storage_.get(), dataset_name, job_name);

      folly::dynamic m = folly::dynamic::object;

      m["id"] = job_id;
      m["name"] = job_name;
      m["featureType"] = "detection";
      auto columns = folly::dynamic::array();
      for (auto column : job_descriptor.columns()) {
        columns.push_back(column.name());
      }
      m["columns"] = columns;

      std::string body = folly::toJson(m);
      response.status(200, "OK").body(body);
    } else if (boost::regex_search(suffix, match_result, features_regex)) {
      suffix = match_result.suffix();

      i32 video_id = std::atoi(match_result[1].str().c_str());

      handle_features(meta, dataset_id, job_id, video_id, suffix, response);
    } else {
      response.status(400, "Bad Request");
      return;
    }
  } else if (path.empty()) {
    // Asking for all job's information
    folly::dynamic json = folly::dynamic::array();

    for (i32 job_id : job_ids) {
      const std::string &job_name = meta.job_names.at(job_id);
      JobDescriptor job_descriptor =
          read_job_descriptor(storage_.get(), dataset_name, job_name);

      folly::dynamic m = folly::dynamic::object;

      m["id"] = job_id;
      m["name"] = meta.job_names.at(job_id);
      m["featureType"] = "detection";
      auto columns = folly::dynamic::array();
      for (auto column : job_descriptor.columns()) {
        columns.push_back(column.name());
      }
      m["columns"] = columns;

      json.push_back(m);
    }
    std::string body = folly::toJson(json);
    response.status(200, "OK").body(body);

  } else {
  }
}

void VideoHandler::handle_features(const DatabaseMetadata &meta, i32 dataset_id,
                                   i32 job_id, i32 video_id,
                                   const std::string &path,
                                   pg::ResponseBuilder &response) {
  if (!(message_->hasQueryParam("start") && message_->hasQueryParam("end") &&
        message_->hasQueryParam("columns"))) {
    response.status(400, "Bad Request");
    return;
  }

  i32 start_frame = message_->getIntQueryParam("start");
  i32 end_frame = message_->getIntQueryParam("end");
  std::string columns_string = message_->getDecodedQueryParam("columns");

  std::vector<std::string> columns = split(columns_string, ',');

  const std::string &dataset_name = meta.dataset_names.at(dataset_id);
  DatasetMetadata dataset_meta{
      read_dataset_descriptor(storage_.get(), dataset_name)};

  i32 stride = 1;
  if (message_->hasQueryParam("stride")) {
    stride = message_->getIntQueryParam("stride");
  }

  f32 threshold = -1;
  if (message_->hasQueryParam("threshold")) {
    threshold = folly::to<f32>(message_->getQueryParam("threshold"));
  }

  // if (message_->has("sampling_filter")) {
  // }

  BBoxParser *parser = new BBoxParser(columns);

  const std::string &job_name = meta.job_names.at(job_id);
  std::string item_name = dataset_meta.item_names().at(video_id);

  VideoMetadata item_meta =
      read_video_metadata(storage_.get(), dataset_name, item_name);
  assert(dataset_meta.type() == DatasetType_Video);

  JobMetadata job_meta(
      dataset_meta.get_descriptor(), {item_meta.get_descriptor()},
      read_job_descriptor(storage_.get(), dataset_name, job_name));

  LoadWorkEntry dummy_entry;
  dummy_entry.interval.start = start_frame;
  dummy_entry.interval.end = end_frame;
  JobMetadata::RowLocations locations =
      job_meta.row_work_item_locations(Sampling::All, video_id, dummy_entry);

  std::vector<std::string> output_names = parser->get_output_names();

  // Get the mapping from frames to timestamps to guide UI toward
  // exact video frame corresponding to the returned features.
  // Needed
  // because some UIs (looking at you html5) do not support frame
  // accurate seeking.
  WebTimestamps timestamps =
      read_web_timestamps(storage_.get(), dataset_name, item_name);

  parser->configure(item_meta);

  folly::dynamic feature_classes = folly::dynamic::array();

  std::vector<std::vector<u8>> output_buffers(output_names.size());
  std::vector<std::vector<i64>> output_buffers_item_sizes(output_names.size());
  std::vector<u64> output_buffers_pos(output_names.size());
  i64 total_frames_in_output_buffer = 0;
  i64 current_frame_in_output_buffer = 0;

  i32 start_difference = 0;
  i32 current_frame = start_frame;
  while (current_frame < end_frame) {
    if (current_frame_in_output_buffer >= total_frames_in_output_buffer) {
      // Find the correct interval
      size_t i;
      for (i = 0; i < locations.work_items.size(); ++i) {
        i32 w_off = i * job_meta.work_item_size();
        if (current_frame >= locations.work_item_intervals[i].start + w_off &&
            current_frame < locations.work_item_intervals[i].end + w_off) {
          break;
        }
      }
      assert(i != locations.work_items.size());
      const auto &interval = locations.work_item_intervals[i];

      i64 start = interval.start + i * job_meta.work_item_size();
      i64 end = interval.end + i * job_meta.work_item_size();

      i32 max_start = std::max((i64)start_frame, start);
      i32 min_end = std::min((i64)end_frame, end);

      for (size_t output_index = 0; output_index < output_names.size();
           ++output_index) {
        std::string output_path = job_item_output_path(
            dataset_name, job_name, item_name, output_names[output_index],
            locations.work_items[i]);

        std::unique_ptr<RandomReadFile> file;
        make_unique_random_read_file(storage_.get(), output_path, file);

        u64 pos = sizeof(u64); // skip past number of rows
        output_buffers_item_sizes[output_index].resize(end - start);
        read(file.get(), (u8 *)output_buffers_item_sizes[output_index].data(),
             (end - start) * sizeof(i64), pos);

        size_t size_left = 0;
        start_difference = current_frame - start;
        for (i32 i = 0; i < start_difference; ++i) {
          pos += output_buffers_item_sizes[output_index][i];
        }

        i64 frames_left = min_end - start;
        for (i32 i = start_difference; i < frames_left; ++i) {
          size_left += output_buffers_item_sizes[output_index][i];
        }

        output_buffers[output_index].resize(size_left);
        read(file.get(), (u8 *)output_buffers[output_index].data(), size_left,
             pos);
        output_buffers_pos[output_index] = 0;
      }

      // Skip past frames at the beginning to fulfill our striding
      current_frame_in_output_buffer = 0;
      total_frames_in_output_buffer = min_end - current_frame;
    }

    // Extract row of feature vectors from output buffers
    std::vector<u8 *> output(output_names.size());
    std::vector<i64> output_sizes(output_names.size());
    for (size_t output_idx = 0; output_idx < output_names.size();
         ++output_idx) {
      size_t size_offset = current_frame_in_output_buffer + start_difference;
      size_t element_size = output_buffers_item_sizes[output_idx][size_offset];
      output_sizes[output_idx] = element_size;

      size_t offset = output_buffers_pos[output_idx];
      output[output_idx] = output_buffers[output_idx].data() + offset;
      output_buffers_pos[output_idx] += element_size;
    }

    // Process vectors
    folly::dynamic feature_data = folly::dynamic::object();
    feature_data["frame"] = current_frame;
    feature_data["time"] = timestamps.pts_timestamps(current_frame) /
                           static_cast<f64>(timestamps.time_base_denominator());
    feature_data["columns"] = folly::dynamic::object();

    parser->parse_output(output, output_sizes, feature_data["columns"]);

    feature_classes.push_back(feature_data);

    current_frame += stride;
    current_frame_in_output_buffer += stride;
  }

  std::string body = folly::toJson(feature_classes);
  response.status(200, "OK").body(body);
}

void VideoHandler::handle_videos(const DatabaseMetadata &meta, i32 dataset_id,
                                 const std::string &path,
                                 pg::ResponseBuilder &response) {
  const std::string &dataset_name = meta.dataset_names.at(dataset_id);
  DatasetMetadata dataset_meta(
      read_dataset_descriptor(storage_.get(), dataset_name));

  auto item_to_json = [](
      i32 item_id, const std::string &video_name, const std::string &media_path,
      const VideoMetadata &item, const WebTimestamps &ts) -> folly::dynamic {
    folly::dynamic meta = folly::dynamic::object;
    meta["id"] = item_id;
    meta["name"] = video_name;
    meta["mediaPath"] = media_path;
    meta["frames"] = item.frames();
    meta["width"] = item.width();
    meta["height"] = item.height();
    folly::dynamic times = folly::dynamic::array();
    for (auto t : ts.pts_timestamps()) {
      times.push_back(t / (f64)ts.time_base_denominator());
    }
    meta["times"] = times;
    return meta;
  };

  folly::dynamic json = folly::dynamic::array();
  boost::smatch match_result;
  if (boost::regex_search(path, match_result, id_regex)) {
    // Asking for a specific videos's information
    i32 video_id = std::atoi(match_result[1].str().c_str());
    if (!(video_id < dataset_meta.item_names().size())) {
      response.status(400, "Bad Request");
      return;
    } else {
      folly::dynamic json = folly::dynamic::object;

      const std::string &video_name = dataset_meta.item_names()[video_id];
      const std::string &media_path = "datasets/" + std::to_string(dataset_id) +
                                      "/media/" + video_name + ".mp4";

      VideoMetadata item =
          read_video_metadata(storage_.get(), dataset_name, video_name);

      WebTimestamps timestamps =
          read_web_timestamps(storage_.get(), dataset_name, video_name);

      json = item_to_json(video_id, video_name, media_path, item, timestamps);
    }
  } else {
    // Requesting all videos metadata
    json = folly::dynamic::array();
    i32 video_id = 0;
    for (const std::string &video_name : dataset_meta.item_names()) {
      const std::string &media_path = "datasets/" + std::to_string(dataset_id) +
                                      "/media/" + video_name + ".mp4";

      VideoMetadata item =
          read_video_metadata(storage_.get(), dataset_name, video_name);

      WebTimestamps timestamps =
          read_web_timestamps(storage_.get(), dataset_name, video_name);

      folly::dynamic meta =
          item_to_json(video_id++, video_name, media_path, item, timestamps);

      json.push_back(meta);
    }
  }

  std::string body = folly::toJson(json);
  response.status(200, "OK").body(body);
}

void VideoHandler::handle_media(const DatabaseMetadata &meta, i32 dataset_id,
                                const std::string &media_path,
                                const std::string &path,
                                pg::ResponseBuilder &response) {
  const pg::HTTPHeaders &headers = message_->getHeaders();
  const std::string &dataset_name = meta.dataset_names.at(dataset_id);
  DatasetDescriptor dataset_descriptor =
      read_dataset_descriptor(storage_.get(), dataset_name);

  const std::string mp4_path =
      dataset_item_video_path(dataset_name, media_path);
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

    const std::string &range_str =
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
  std::unique_ptr<folly::IOBuf> buffer{folly::IOBuf::createCombined(read_size)};
  buffer->append(read_size);
  size_t size_read;

  std::cout << "reading request " << message_->getPath() << std::endl;
  fflush(stdout);
  u64 pos = byte_range_start;
  read(file.get(), buffer->writableData(), read_size, pos);
  std::cout << "finished reading request " << message_->getPath() << std::endl;
  fflush(stdout);

  response.status(206, "Partial Content")
      .header(pg::HTTP_HEADER_CONTENT_TYPE, mime_type)
      .header(pg::HTTP_HEADER_ACCEPT_RANGES, "bytes")
      .header(pg::HTTP_HEADER_CONTENT_RANGE,
              "bytes " + std::to_string(byte_range_start) + "-" +
                  std::to_string(byte_range_end) + "/" +
                  std::to_string(file_size))
      .body(std::move(buffer));
}
}
