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

#include "scanner/util/db.h"
#include "scanner/engine.h"
#include "scanner/util/jsoncpp.h"
#include "scanner/util/storehouse.h"
#include "scanner/util/util.h"
#include "storehouse/storage_backend.h"

#include <errno.h>
#include <libgen.h>
#include <limits.h> /* PATH_MAX */
#include <string.h>
#include <sys/stat.h> /* mkdir(2) */
#include <cassert>
#include <cstdarg>
#include <iostream>
#include <sstream>

using storehouse::WriteFile;
using storehouse::RandomReadFile;
using storehouse::StoreResult;

namespace scanner {

DatabaseMetadata::DatabaseMetadata() : next_dataset_id(0), next_job_id(0) {}

DatabaseMetadata::DatabaseMetadata(const DatabaseDescriptor& d)
    : descriptor(d),
      next_dataset_id(d.next_dataset_id()),
      next_job_id(d.next_job_id()) {
  for (int i = 0; i < descriptor.datasets_size(); ++i) {
    const DatabaseDescriptor_Dataset& dataset = descriptor.datasets(i);
    dataset_names.insert({dataset.id(), dataset.name()});
    dataset_job_ids[dataset.id()] = {};
  }
  for (int i = 0; i < descriptor.jobs_size(); ++i) {
    const DatabaseDescriptor_Job& job = descriptor.jobs(i);
    job_names.insert({job.id(), job.name()});
  }
  for (int i = 0; i < descriptor.job_to_datasets_size(); ++i) {
    const DatabaseDescriptor_JobToDataset& job_to_dataset =
        descriptor.job_to_datasets(i);
    dataset_job_ids[job_to_dataset.dataset_id()].insert(
        job_to_dataset.job_id());
  }
}

const DatabaseDescriptor& DatabaseMetadata::get_descriptor() const {
  descriptor.set_next_dataset_id(next_dataset_id);
  descriptor.set_next_job_id(next_job_id);
  descriptor.clear_datasets();
  descriptor.clear_jobs();
  descriptor.clear_job_to_datasets();

  for (auto& kv : dataset_names) {
    auto dataset = descriptor.add_datasets();
    dataset->set_id(kv.first);
    dataset->set_name(kv.second);
  }

  for (auto& kv : job_names) {
    auto job = descriptor.add_jobs();
    job->set_id(kv.first);
    job->set_name(kv.second);
  }

  for (auto& kv : dataset_job_ids) {
    for (i32 job_id : kv.second) {
      auto job = descriptor.add_job_to_datasets();
      job->set_dataset_id(kv.first);
      job->set_job_id(job_id);
    }
  }

  return descriptor;
}

bool DatabaseMetadata::has_dataset(const std::string& dataset) const {
  for (const auto& kv : dataset_names) {
    if (kv.second == dataset) {
      return true;
    }
  }
  return false;
}

bool DatabaseMetadata::has_dataset(i32 dataset_id) const {
  return dataset_names.count(dataset_id) > 0;
}

i32 DatabaseMetadata::get_dataset_id(const std::string& dataset) const {
  i32 id = -1;
  for (const auto& kv : dataset_names) {
    if (kv.second == dataset) {
      id = kv.first;
      break;
    }
  }
  assert(id != -1);
  return id;
}

const std::string& DatabaseMetadata::get_dataset_name(i32 dataset_id) const {
  return dataset_names.at(dataset_id);
}

i32 DatabaseMetadata::add_dataset(const std::string& dataset) {
  i32 dataset_id = next_dataset_id++;
  dataset_names[dataset_id] = dataset;
  dataset_job_ids[dataset_id] = {};
  return dataset_id;
}

void DatabaseMetadata::remove_dataset(i32 dataset_id) {
  for (i32 job_id : dataset_job_ids.at(dataset_id)) {
    job_names.erase(job_id);
  }
  assert(dataset_job_ids.count(dataset_id) > 0);
  assert(dataset_names.count(dataset_id) > 0);
  dataset_job_ids.erase(dataset_id);
  dataset_names.erase(dataset_id);
}

bool DatabaseMetadata::has_job(i32 dataset_id, const std::string& job) const {
  for (i32 job_id : dataset_job_ids.at(dataset_id)) {
    if (job == job_names.at(job_id)) {
      return true;
    }
  }
  return false;
}

bool DatabaseMetadata::has_job(i32 job_id) const {
  return job_names.count(job_id) > 0;
}

i32 DatabaseMetadata::get_job_id(i32 dataset_id, const std::string& job) const {
  i32 job_id = -1;
  for (i32 j_id : dataset_job_ids.at(dataset_id)) {
    if (job == job_names.at(j_id)) {
      job_id = j_id;
      break;
    }
  }
  assert(job_id != -1);
  return job_id;
}

const std::string& DatabaseMetadata::get_job_name(i32 job_id) const {
  return job_names.at(job_id);
}

i32 DatabaseMetadata::add_job(i32 dataset_id, const std::string& job_name) {
  i32 job_id = next_job_id++;
  dataset_job_ids.at(dataset_id).insert(job_id);
  job_names[job_id] = job_name;
  return job_id;
}

void DatabaseMetadata::remove_job(i32 job_id) {
  bool found = false;
  for (auto& kv : dataset_job_ids) {
    if (kv.second.count(job_id) > 0) {
      kv.second.erase(job_id);
      found = true;
      break;
    }
  }
  assert(found);
  assert(job_names.count(job_id) > 0);
  job_names.erase(job_id);
}

///////////////////////////////////////////////////////////////////////////////
/// DatasetMetadata
DatasetMetadata::DatasetMetadata() {}

DatasetMetadata::DatasetMetadata(const DatasetDescriptor& descriptor)
    : descriptor(descriptor) {}

const DatasetDescriptor& DatasetMetadata::get_descriptor() const {
  return descriptor;
}

i32 DatasetMetadata::id() const { return descriptor.id(); }

std::string DatasetMetadata::name() const { return descriptor.name(); }

DatasetType DatasetMetadata::type() const { return descriptor.type(); }

i32 DatasetMetadata::total_frames() const {
  if (this->type() == DatasetType_Video) {
    return descriptor.video_data().total_frames();
  } else if (this->type() == DatasetType_Image) {
    return descriptor.image_data().total_images();
  } else {
    assert(false);
    return {};
  }
}

i32 DatasetMetadata::max_width() const {
  if (this->type() == DatasetType_Video) {
    return descriptor.video_data().max_width();
  } else if (this->type() == DatasetType_Image) {
    return descriptor.image_data().max_width();
  } else {
    assert(false);
    return {};
  }
}

i32 DatasetMetadata::max_height() const {
  if (this->type() == DatasetType_Video) {
    return descriptor.video_data().max_height();
  } else if (this->type() == DatasetType_Image) {
    return descriptor.image_data().max_height();
  } else {
    assert(false);
    return {};
  }
}

std::vector<std::string> DatasetMetadata::original_paths() const {
  if (this->type() == DatasetType_Video) {
    return std::vector<std::string>(
        descriptor.video_data().original_video_paths().begin(),
        descriptor.video_data().original_video_paths().end());
  } else if (this->type() == DatasetType_Image) {
    return std::vector<std::string>(
        descriptor.image_data().original_image_paths().begin(),
        descriptor.image_data().original_image_paths().end());
  } else {
    assert(false);
    return {};
  }
}

std::vector<std::string> DatasetMetadata::item_names() const {
  if (this->type() == DatasetType_Video) {
    return std::vector<std::string>(
        descriptor.video_data().video_names().begin(),
        descriptor.video_data().video_names().end());
  } else if (this->type() == DatasetType_Image) {
    std::vector<std::string> item_names;
    for (i32 i = 0; i < descriptor.image_data().format_groups_size(); ++i) {
      item_names.push_back(std::to_string(i));
    }
    return item_names;
  } else {
    assert(false);
    return {};
  }
}

///////////////////////////////////////////////////////////////////////////////
/// VideoMetdata
VideoMetadata::VideoMetadata() {}

VideoMetadata::VideoMetadata(const VideoDescriptor& descriptor)
    : descriptor(descriptor) {}

const VideoDescriptor& VideoMetadata::get_descriptor() const {
  return descriptor;
}

i32 VideoMetadata::frames() const { return descriptor.frames(); }

i32 VideoMetadata::width() const { return descriptor.width(); }

i32 VideoMetadata::height() const { return descriptor.height(); }

std::vector<i64> VideoMetadata::keyframe_positions() const {
  return std::vector<i64>(descriptor.keyframe_positions().begin(),
                          descriptor.keyframe_positions().end());
}

std::vector<i64> VideoMetadata::keyframe_byte_offsets() const {
  return std::vector<i64>(descriptor.keyframe_byte_offsets().begin(),
                          descriptor.keyframe_byte_offsets().end());
}

///////////////////////////////////////////////////////////////////////////////
/// ImageFormatGroupMetadata
ImageFormatGroupMetadata::ImageFormatGroupMetadata() {}

ImageFormatGroupMetadata::ImageFormatGroupMetadata(
    const ImageFormatGroupDescriptor& descriptor)
    : descriptor(descriptor) {}

const ImageFormatGroupDescriptor& ImageFormatGroupMetadata::get_descriptor()
    const {
  return descriptor;
}

i32 ImageFormatGroupMetadata::num_images() const {
  return descriptor.num_images();
}

i32 ImageFormatGroupMetadata::width() const { return descriptor.width(); }

i32 ImageFormatGroupMetadata::height() const { return descriptor.height(); }

ImageEncodingType ImageFormatGroupMetadata::encoding_type() const {
  return descriptor.encoding_type();
}

ImageColorSpace ImageFormatGroupMetadata::color_space() const {
  return descriptor.color_space();
}

std::vector<i64> ImageFormatGroupMetadata::compressed_sizes() const {
  return std::vector<i64>(descriptor.compressed_sizes().begin(),
                          descriptor.compressed_sizes().end());
}

///////////////////////////////////////////////////////////////////////////////
/// JobMetadata
JobMetadata::JobMetadata() : complete_(false) {}
JobMetadata::JobMetadata(const DatasetDescriptor& dataset,
                         const std::vector<VideoDescriptor>& videos,
                         const JobDescriptor& job)
    : dataset_descriptor(dataset),
      video_descriptors(videos),
      job_descriptor(job),
      complete_(true) {
  std::set<i32> provided_ids;
  for (const VideoDescriptor& desc : videos) {
    provided_ids.insert(desc.id());
  }
  for (i32 id : dataset.video_data().video_ids()) {
    if (provided_ids.count(id) == 0) {
      complete_ = false;
    }
  }
}
JobMetadata::JobMetadata(const DatasetDescriptor& dataset,
                         const std::vector<ImageFormatGroupDescriptor>& images,
                         const JobDescriptor& job)
    : dataset_descriptor(dataset),
      format_descriptors(images),
      job_descriptor(job),
      complete_(true) {
  std::set<i32> provided_ids;
  for (const ImageFormatGroupDescriptor& desc : images) {
    provided_ids.insert(desc.id());
  }
  for (i32 id : dataset.image_data().format_group_ids()) {
    if (provided_ids.count(id) == 0) {
      complete_ = false;
    }
  }
}

const DatasetDescriptor& JobMetadata::get_dataset_descriptor() const {
  return dataset_descriptor;
}

const JobDescriptor& JobMetadata::get_job_descriptor() const {
  return job_descriptor;
}

i32 JobMetadata::id() const { return job_descriptor.id(); }

std::string JobMetadata::name() const { return job_descriptor.name(); }

std::vector<std::string> JobMetadata::columns() const {
  std::vector<std::string> columns = {"frame"};
  for (auto& c : job_descriptor.columns()) {
    columns.push_back(c.name());
  }
  return columns;
}

Sampling JobMetadata::sampling() const {
  Sampling sampling;
  switch (job_descriptor.sampling()) {
    case JobDescriptor::All:
      sampling = Sampling::All;
      break;
    case JobDescriptor::Strided:
      sampling = Sampling::Strided;
      break;
    case JobDescriptor::Gather:
      sampling = Sampling::Gather;
      break;
    case JobDescriptor::SequenceGather:
      sampling = Sampling::SequenceGather;
      break;
  }
  return sampling;
}

i32 JobMetadata::work_item_size() const {
  return job_descriptor.work_item_size();
}

i32 JobMetadata::total_rows() const {
  Sampling sampling = this->sampling();
  i32 rows = 0;

  std::vector<i32> total_frames_per_item = rows_per_item();
  switch (sampling) {
    case Sampling::All: {
      for (i32 f : total_frames_per_item) {
        rows += f;
      }
      break;
    }
    case Sampling::Strided: {
      for (i32 f : total_frames_per_item) {
        rows += f / job_descriptor.stride();
      }
      break;
    }
    case Sampling::Gather: {
      for (const auto& samples : job_descriptor.gather_points()) {
        rows += samples.frames_size();
      }
      break;
    }
    case Sampling::SequenceGather: {
      for (const auto& samples : job_descriptor.gather_sequences()) {
        for (const JobDescriptor::StridedInterval& interval :
             samples.intervals()) {
          rows += (interval.end() - interval.start()) / interval.stride();
        }
      }
      break;
    }
  }
  return rows;
}

std::vector<JobMetadata::GroupSample> JobMetadata::sampled_frames() const {
  Sampling sampling = this->sampling();
  std::vector<GroupSample> group_samples;

  std::vector<i32> total_frames_per_item = rows_per_item();
  switch (sampling) {
    case Sampling::All: {
      for (size_t i = 0; i < total_frames_per_item.size(); ++i) {
        group_samples.emplace_back();
        GroupSample& s = group_samples.back();
        s.group_index = static_cast<i32>(i);
        i32 tot_frames = total_frames_per_item[i];
        for (i32 f = 0; f < tot_frames; ++f) {
          s.frames.push_back(f);
        }
      }
      break;
    }
    case Sampling::Strided: {
      i32 stride = job_descriptor.stride();
      for (size_t i = 0; i < total_frames_per_item.size(); ++i) {
        group_samples.emplace_back();
        GroupSample& s = group_samples.back();
        s.group_index = static_cast<i32>(i);
        i32 tot_frames = total_frames_per_item[i];
        for (i32 f = 0; f < tot_frames; f += stride) {
          s.frames.push_back(f);
        }
      }
      break;
    }
    case Sampling::Gather: {
      for (const auto& samples : job_descriptor.gather_points()) {
        group_samples.emplace_back();
        GroupSample& s = group_samples.back();
        s.group_index = samples.video_index();
        for (i32 f : samples.frames()) {
          s.frames.push_back(f);
        }
      }
      break;
    }
    case Sampling::SequenceGather: {
      for (const auto& samples : job_descriptor.gather_sequences()) {
        group_samples.emplace_back();
        GroupSample& s = group_samples.back();
        s.group_index = samples.video_index();
        for (const JobDescriptor::StridedInterval& interval :
             samples.intervals()) {
          for (i32 f = interval.start(); f < interval.end();
               f += interval.stride()) {
            s.frames.push_back(f);
          }
        }
      }
      break;
    }
  }
  return group_samples;
}

JobMetadata::RowLocations JobMetadata::row_work_item_locations(
    Sampling sampling, i32 group_id, const LoadWorkEntry& entry) const {
  RowLocations locations;
  std::vector<i32>& items = locations.work_items;
  std::vector<Interval>& intervals = locations.work_item_intervals;

  i32 total_rows = rows_in_item(group_id);
  switch (sampling) {
    case Sampling::All: {
      i32 start = entry.interval.start;
      i32 end = entry.interval.end;
      i32 tot_frames = total_rows;
      i32 work_item_size = this->work_item_size();
      i32 first_work_item = start / work_item_size;
      i32 first_start_row = start % work_item_size;
      i32 frames_left_in_wi = work_item_size - first_start_row;
      items.push_back(first_work_item);
      intervals.push_back(
          Interval{first_start_row, std::min(work_item_size, (end - start))});
      start += work_item_size;
      i32 curr_work_item = first_work_item + 1;
      while (start < end) {
        items.push_back(curr_work_item);
        intervals.push_back(
            Interval{0, std::min(work_item_size, (end - start))});
        start += work_item_size;
        curr_work_item++;
      }
      break;
    }
    case Sampling::Strided:
    case Sampling::Gather:
    case Sampling::SequenceGather: {
      assert(false);
      break;
    }
  }
  return locations;
}

JobMetadata::FrameLocations JobMetadata::frame_locations(
    Sampling sampling, i32 video_index, const LoadWorkEntry& entry) const {
  Sampling job_sampling = this->sampling();
  FrameLocations locations;
  std::vector<Interval>& intervals = locations.intervals;
  std::vector<DecodeArgs>& dargs = locations.video_args;
  std::vector<ImageDecodeArgs>& image_dargs = locations.image_args;

  if (job_sampling == Sampling::All) {
    if (sampling == Sampling::All) {
      intervals.push_back(Interval{entry.interval.start, entry.interval.end});

      // Video decode arguments
      DecodeArgs decode_args;
      decode_args.set_sampling(DecodeArgs::All);
      decode_args.mutable_interval()->set_start(entry.interval.start);
      decode_args.mutable_interval()->set_end(entry.interval.end);
      decode_args.mutable_interval()->set_stride(0);  // Dummy

      dargs.push_back(decode_args);

      ImageDecodeArgs image_args;
      image_args.set_sampling(ImageDecodeArgs::All);
      image_args.mutable_interval()->set_start(entry.interval.start);
      image_args.mutable_interval()->set_end(entry.interval.end);
      image_args.mutable_interval()->set_stride(0);  // Dummy

      image_dargs.push_back(image_args);

    } else if (sampling == Sampling::Strided) {
      // TODO(apoms): loading a consecutive portion of the video stream might
      //   be inefficient if the stride is much larger than a single GOP.
      intervals.push_back(
          Interval{entry.strided_interval.start, entry.strided_interval.end});

      // Video decode arguments
      DecodeArgs decode_args;
      decode_args.set_sampling(DecodeArgs::Strided);
      decode_args.mutable_interval()->set_start(entry.strided_interval.start);
      decode_args.mutable_interval()->set_end(entry.strided_interval.end);
      decode_args.mutable_interval()->set_stride(entry.strided_interval.stride);
      decode_args.set_stride(entry.strided_interval.stride);

      dargs.push_back(decode_args);

      ImageDecodeArgs image_args;
      image_args.set_sampling(ImageDecodeArgs::Strided);
      image_args.mutable_interval()->set_start(entry.strided_interval.start);
      image_args.mutable_interval()->set_end(entry.strided_interval.end);
      image_args.mutable_interval()->set_stride(entry.strided_interval.stride);
      image_args.set_stride(entry.strided_interval.stride);

      image_dargs.push_back(image_args);

    } else if (sampling == Sampling::Gather) {
      // TODO(apoms): This implementation is not efficient for gathers which
      //   overlap in the same GOP.
      for (size_t i = 0; i < entry.gather_points.size(); ++i) {
        intervals.push_back(
            Interval{entry.gather_points[i], entry.gather_points[i] + 1});

        // Video decode arguments
        DecodeArgs decode_args;
        decode_args.set_sampling(DecodeArgs::Gather);
        decode_args.add_gather_points(entry.gather_points[i]);

        dargs.push_back(decode_args);
      }
    } else if (sampling == Sampling::SequenceGather) {
      for (const StridedInterval& s : entry.gather_sequences) {
        intervals.push_back(Interval{s.start, s.end});
      }

      for (size_t i = 0; i < entry.gather_sequences.size(); ++i) {
        // Video decode arguments
        DecodeArgs decode_args;
        decode_args.set_sampling(DecodeArgs::SequenceGather);
        DecodeArgs::StridedInterval* intvl = decode_args.add_gather_sequences();
        intvl->set_start(entry.gather_sequences[i].start);
        intvl->set_end(entry.gather_sequences[i].end);
        intvl->set_stride(entry.gather_sequences[i].stride);

        dargs.push_back(decode_args);
      }
    }
  } else {
    // Only support all sampling on derived datasets
    assert(sampling == Sampling::All);

    std::vector<GroupSample> samples = this->sampled_frames();
    GroupSample video_sample;
    video_sample.group_index = -1;
    for (GroupSample& s : samples) {
      if (s.group_index == video_index) {
        video_sample = s;
        break;
      }
    }
    assert(video_sample.group_index != -1);

    i32 start = entry.interval.start;
    i32 end = entry.interval.end;
    if (job_sampling == Sampling::Strided) {
      i32 stride = job_descriptor.stride();
      i32 stride_start = video_sample.frames[start];
      i32 stride_end = video_sample.frames[end - 1] + stride;
      intervals.push_back(Interval{stride_start, stride_end});

      // Video decode arguments
      DecodeArgs decode_args;
      decode_args.set_sampling(DecodeArgs::Strided);
      decode_args.mutable_interval()->set_start(stride_start);
      decode_args.mutable_interval()->set_end(stride_end);
      decode_args.set_stride(stride);

      dargs.push_back(decode_args);

      ImageDecodeArgs image_args;
      image_args.set_sampling(ImageDecodeArgs::All);
      image_args.mutable_interval()->set_start(stride_start);
      image_args.mutable_interval()->set_end(stride_end);
      decode_args.set_stride(stride);

      image_dargs.push_back(image_args);

    } else if (job_sampling == Sampling::Gather) {
      for (i32 s = start; s < end; ++s) {
        i32 frame = video_sample.frames[s];
        intervals.push_back(Interval{frame, frame + 1});

        // Video decode arguments
        DecodeArgs decode_args;
        decode_args.set_sampling(DecodeArgs::Gather);
        decode_args.add_gather_points(frame);

        dargs.push_back(decode_args);
      }
    } else if (job_sampling == Sampling::SequenceGather) {
      size_t s_idx = 0;
      size_t i_idx = 0;
      i32 frames_so_far = 0;
      JobDescriptor::SequenceSamples* sample =
          job_descriptor.mutable_gather_sequences(s_idx);
      while (start < end) {
        while (sample->video_index() != video_index) {
          sample = job_descriptor.mutable_gather_sequences(++s_idx);
          i_idx = 0;
          frames_so_far = 0;
        }
        JobDescriptor::StridedInterval* interval =
            sample->mutable_intervals(i_idx);
        i32 stride = interval->stride();
        i32 interval_offset = 0;
        while (frames_so_far < start) {
          i32 needed_frames = start - frames_so_far;
          i32 frames_in_interval =
              (interval->end() - interval->start()) / stride;
          if (frames_in_interval <= needed_frames) {
            interval = sample->mutable_intervals(++i_idx);
            interval_offset = 0;
          } else {
            interval_offset = needed_frames * stride;
          }
          frames_so_far += std::min(frames_in_interval, needed_frames);
        }
        i32 start_frame = interval->start() + interval_offset;
        i32 frames_left_in_interval = (interval->end() - start_frame) / stride;
        i32 end_frame = start_frame +
                        std::min(frames_left_in_interval * stride, end - start);
        start += frames_left_in_interval;
        frames_so_far += frames_left_in_interval;

        intervals.push_back(Interval{start_frame, end_frame});

        // Video decode arguments
        DecodeArgs decode_args;
        decode_args.set_sampling(DecodeArgs::SequenceGather);
        DecodeArgs::StridedInterval* intvl = decode_args.add_gather_sequences();
        intvl->set_start(start_frame);
        intvl->set_end(end_frame);
        intvl->set_stride(stride);

        dargs.push_back(decode_args);
      }
    }
  }
  return locations;
}

std::vector<i32> JobMetadata::rows_per_item() const {
  assert(complete_);
  std::vector<i32> total_frames_per_item;
  if (dataset_descriptor.type() == DatasetType_Video) {
    for (const VideoDescriptor& meta : video_descriptors) {
      total_frames_per_item.push_back(meta.frames());
    }
  } else if (dataset_descriptor.type() == DatasetType_Image) {
    for (const ImageFormatGroupDescriptor& meta : format_descriptors) {
      total_frames_per_item.push_back(meta.num_images());
    }
  }
  return total_frames_per_item;
}

i32 JobMetadata::rows_in_item(i32 group_id) const {
  i32 total_rows = -1;
  if (dataset_descriptor.type() == DatasetType_Video) {
    for (const VideoDescriptor& meta : video_descriptors) {
      if (meta.id() == group_id) {
        total_rows = meta.frames();
        break;
      }
    }
  } else if (dataset_descriptor.type() == DatasetType_Image) {
    for (const ImageFormatGroupDescriptor& meta : format_descriptors) {
      if (meta.id() == group_id) {
        total_rows = meta.num_images();
        break;
      }
    }
  }
  assert(total_rows != -1);
  return total_rows;
}

namespace {

template <typename T>
void serialize(storehouse::WriteFile* file, const T& descriptor) {
  int size = descriptor.ByteSize();
  std::vector<u8> data(size);
  descriptor.SerializeToArray(data.data(), size);
  write(file, data.data(), size);
}

template <typename T>
T deserialize(storehouse::RandomReadFile* file, u64& pos) {
  T descriptor;
  std::vector<u8> data = storehouse::read_entire_file(file, pos);
  descriptor.ParseFromArray(data.data(), data.size());
  return descriptor;
}
}

void serialize_database_metadata(storehouse::WriteFile* file,
                                 const DatabaseMetadata& metadata) {
  serialize(file, metadata.get_descriptor());
}

DatabaseMetadata deserialize_database_metadata(storehouse::RandomReadFile* file,
                                               u64& pos) {
  return DatabaseMetadata(deserialize<DatabaseDescriptor>(file, pos));
}

void serialize_dataset_descriptor(WriteFile* file,
                                  const DatasetDescriptor& descriptor) {
  serialize(file, descriptor);
}

DatasetDescriptor deserialize_dataset_descriptor(RandomReadFile* file,
                                                 uint64_t& pos) {
  return deserialize<DatasetDescriptor>(file, pos);
}

void serialize_video_metadata(WriteFile* file, const VideoMetadata& metadata) {
  serialize(file, metadata.get_descriptor());
}

VideoMetadata deserialize_video_metadata(RandomReadFile* file, uint64_t& pos) {
  return VideoMetadata{deserialize<VideoDescriptor>(file, pos)};
}

void serialize_image_format_group_metadata(
    WriteFile* file, const ImageFormatGroupMetadata& metadata) {
  serialize(file, metadata.get_descriptor());
}

ImageFormatGroupMetadata deserialize_image_format_group_metadata(
    RandomReadFile* file, uint64_t& pos) {
  return ImageFormatGroupMetadata{
      deserialize<ImageFormatGroupDescriptor>(file, pos)};
}

void serialize_web_timestamps(WriteFile* file, const WebTimestamps& ts) {
  serialize(file, ts);
}

WebTimestamps deserialize_web_timestamps(RandomReadFile* file, uint64_t& pos) {
  return deserialize<WebTimestamps>(file, pos);
}

void serialize_job_descriptor(WriteFile* file,
                              const JobDescriptor& descriptor) {
  serialize(file, descriptor);
}

JobDescriptor deserialize_job_descriptor(RandomReadFile* file, uint64_t& pos) {
  return deserialize<JobDescriptor>(file, pos);
}

std::string PREFIX = "";

void set_database_path(std::string path) { PREFIX = path + "/"; }
}
