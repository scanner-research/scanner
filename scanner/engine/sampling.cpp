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

#include "scanner/engine/sampling.h"

namespace scanner {

std::vector<GroupSample> sampled_frames() const {
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

RowLocations row_work_item_locations(
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

FrameLocations frame_locations(
    i32 video_index, const LoadWorkEntry& entry) const {
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
      image_args.set_sampling(ImageDecodeArgs::All);
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
