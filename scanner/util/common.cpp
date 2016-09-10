#include "scanner/util/common.h"
#include "scanner/util/util.h"
#include "storehouse/storage_backend.h"

#include <cassert>
#include <cstdarg>
#include <sstream>
#include <limits.h>     /* PATH_MAX */
#include <sys/stat.h>   /* mkdir(2) */
#include <string.h>
#include <errno.h>
#include <libgen.h>

using storehouse::WriteFile;
using storehouse::RandomReadFile;
using storehouse::StoreResult;

namespace scanner {

int CPUS_PER_NODE = 1;           // Number of available CPUs per node
int GPUS_PER_NODE = 1;           // Number of available GPUs per node
int GLOBAL_BATCH_SIZE = 64;      // Batch size for network
int BATCHES_PER_WORK_ITEM = 4;   // How many batches per work item
int TASKS_IN_QUEUE_PER_GPU = 4;  // How many tasks per GPU to allocate to a node
int LOAD_WORKERS_PER_NODE = 2;   // Number of worker threads loading data
int SAVE_WORKERS_PER_NODE = 2;   // Number of worker threads loading data
int NUM_CUDA_STREAMS = 32;       // Number of cuda streams for image processing


void serialize_dataset_descriptor(
  WriteFile* file,
  const DatasetDescriptor& descriptor)
{
  StoreResult result;
  // Number of videos
  size_t num_videos = descriptor.original_video_paths.size();

  EXP_BACKOFF(
    file->append(sizeof(size_t), reinterpret_cast<char*>(&num_videos)),
    result);
  exit_on_error(result);

  for (size_t i = 0; i < num_videos; ++i) {
    const std::string& path = descriptor.original_video_paths[i];
    EXP_BACKOFF(
      file->append(path.size() + 1, path.c_str()),
      result);
    exit_on_error(result);

    const std::string& item_name = descriptor.item_names[i];
    EXP_BACKOFF(
      file->append(item_name.size() + 1, item_name.c_str()),
      result);
    exit_on_error(result);
  }
}

DatasetDescriptor deserialize_dataset_descriptor(
  RandomReadFile* file,
  uint64_t& pos)
{
  StoreResult result;
  size_t size_read;

  DatasetDescriptor descriptor;

  // Size of metadata
  size_t num_videos;
  EXP_BACKOFF(
    file->read(pos,
               sizeof(size_t),
               reinterpret_cast<char*>(&num_videos),
               size_read),
    result);
  exit_on_error(result);
  assert(size_read == sizeof(size_t));
  pos += size_read;

  // Read all data into buffer
  std::vector<char> bytes = read_entire_file(file, pos);
  char* buffer = bytes.data();
  for (size_t i = 0; i < num_videos; ++i) {
    descriptor.original_video_paths.emplace_back(buffer);
    buffer += descriptor.original_video_paths[i].size() + 1;

    descriptor.item_names.emplace_back(buffer);
    buffer += descriptor.item_names[i].size() + 1;
  }
  return descriptor;
}

void serialize_dataset_item_metadata(
  WriteFile* file,
  const DatasetItemMetadata& metadata)
{
  // Frames
  StoreResult result;
  EXP_BACKOFF(
    file->append(sizeof(int32_t),
                 reinterpret_cast<const char*>(&metadata.frames)),
    result);
  exit_on_error(result);

  // Width
  EXP_BACKOFF(
    file->append(sizeof(int32_t),
                 reinterpret_cast<const char*>(&metadata.width)),
    result);
  exit_on_error(result);

  // Height
  EXP_BACKOFF(
    file->append(sizeof(int32_t),
                 reinterpret_cast<const char*>(&metadata.height)),
    result);
  exit_on_error(result);

  // Codec type
  EXP_BACKOFF(
    file->append(sizeof(cudaVideoCodec),
                 reinterpret_cast<const char*>(&metadata.codec_type)),
    result);
  exit_on_error(result);

  // Chroma format
  EXP_BACKOFF(
    file->append(sizeof(cudaVideoChromaFormat),
                 reinterpret_cast<const char*>(&metadata.chroma_format)),
    result);
  exit_on_error(result);

  // Size of metadata
  size_t metadata_packets_size = metadata.metadata_packets.size();
  EXP_BACKOFF(
    file->append(sizeof(size_t),
                 reinterpret_cast<const char*>(
                   &metadata_packets_size)),
    result);
  exit_on_error(result);

  // Metadata packets
  EXP_BACKOFF(
    file->append(metadata_packets_size,
                 reinterpret_cast<const char*>(
                   metadata.metadata_packets.data())),
    result);
  exit_on_error(result);

  // Keyframe info
  assert(metadata.keyframe_positions.size() ==
         metadata.keyframe_timestamps.size());
  assert(metadata.keyframe_positions.size() ==
         metadata.keyframe_byte_offsets.size());

  size_t num_keyframes = metadata.keyframe_positions.size();

  EXP_BACKOFF(
    file->append(sizeof(size_t), reinterpret_cast<char*>(&num_keyframes)),
    result);
  exit_on_error(result);

  EXP_BACKOFF(
    file->append(sizeof(int64_t) * num_keyframes,
                 reinterpret_cast<const char*>(
                   metadata.keyframe_positions.data())),
    result);
  exit_on_error(result);

  EXP_BACKOFF(
    file->append(sizeof(int64_t) * num_keyframes,
                 reinterpret_cast<const char*>(
                   metadata.keyframe_timestamps.data())),
    result);
  exit_on_error(result);

  EXP_BACKOFF(
    file->append(sizeof(int64_t) * num_keyframes,
                 reinterpret_cast<const char*>(
                   metadata.keyframe_byte_offsets.data())),
    result);
  exit_on_error(result);
}

DatasetItemMetadata deserialize_dataset_item_metadata(
  RandomReadFile* file,
  uint64_t& pos)
{
  StoreResult result;
  size_t size_read;

  DatasetItemMetadata meta;
  // Frames
  EXP_BACKOFF(
    file->read(pos,
               sizeof(int32_t),
               reinterpret_cast<char*>(&meta.frames),
               size_read),
    result);
  exit_on_error(result);
  assert(size_read == sizeof(int32_t));
  pos += size_read;

  // Width
  EXP_BACKOFF(
    file->read(pos,
               sizeof(int32_t),
               reinterpret_cast<char*>(&meta.width),
               size_read),
    result);
  exit_on_error(result);
  assert(size_read == sizeof(int32_t));
  pos += size_read;

  // Height
  EXP_BACKOFF(
    file->read(pos,
               sizeof(int32_t),
               reinterpret_cast<char*>(&meta.height),
               size_read),
    result);
  exit_on_error(result);
  assert(size_read == sizeof(int32_t));
  pos += size_read;

  // Codec type
  EXP_BACKOFF(
    file->read(pos,
               sizeof(cudaVideoCodec),
               reinterpret_cast<char*>(&meta.codec_type),
               size_read),
    result);
  exit_on_error(result);
  assert(size_read == sizeof(cudaVideoCodec));
  pos += size_read;

  // Chroma format
  EXP_BACKOFF(
    file->read(pos,
               sizeof(cudaVideoChromaFormat),
               reinterpret_cast<char*>(&meta.chroma_format),
               size_read),
    result);
  exit_on_error(result);
  assert(size_read == sizeof(cudaVideoChromaFormat));
  pos += size_read;

  // Size of metadata
  size_t metadata_size;
  EXP_BACKOFF(
    file->read(pos,
               sizeof(size_t),
               reinterpret_cast<char*>(&metadata_size),
               size_read),
    result);
  exit_on_error(result);
  assert(size_read == sizeof(size_t));
  pos += size_read;

  // Metadata packets
  meta.metadata_packets.resize(metadata_size);
  EXP_BACKOFF(
    file->read(pos,
               metadata_size,
               reinterpret_cast<char*>(meta.metadata_packets.data()),
               size_read),
    result);
  exit_on_error(result);
  assert(size_read == metadata_size);
  pos += size_read;

  size_t num_keyframes;
  // HACK(apoms): Reading just a single size_t is inefficient because
  //              the file interface does not buffer or preemptively fetch
  //              a larger block of data to amortize network overheads. We
  //              should instead read the entire file into a buffer because we
  //              know it is fairly small and then deserialize from there.
  EXP_BACKOFF(
    file->read(pos,
               sizeof(size_t),
               reinterpret_cast<char*>(&num_keyframes),
               size_read),
    result);
  exit_on_error(result);
  assert(size_read == sizeof(size_t));
  pos += size_read;

  meta.keyframe_positions.resize(num_keyframes);
  meta.keyframe_timestamps.resize(num_keyframes);
  meta.keyframe_byte_offsets.resize(num_keyframes);

  EXP_BACKOFF(
    file->read(pos,
               sizeof(int64_t) * num_keyframes,
               reinterpret_cast<char*>(meta.keyframe_positions.data()),
               size_read),
    result);
  exit_on_error(result);
  assert(size_read == sizeof(int64_t) * num_keyframes);
  pos += size_read;

  EXP_BACKOFF(
    file->read(pos,
               sizeof(int64_t) * num_keyframes,
               reinterpret_cast<char*>(meta.keyframe_timestamps.data()),
               size_read),
    result);
  assert(result == StoreResult::Success ||
         result == StoreResult::EndOfFile);
  assert(size_read == sizeof(int64_t) * num_keyframes);
  pos += size_read;

  EXP_BACKOFF(
    file->read(pos,
               sizeof(int64_t) * num_keyframes,
               reinterpret_cast<char*>(meta.keyframe_byte_offsets.data()),
               size_read),
    result);
  assert(result == StoreResult::Success ||
         result == StoreResult::EndOfFile);
  assert(size_read == sizeof(int64_t) * num_keyframes);
  pos += size_read;

  return meta;
}

void serialize_dataset_item_web_timestamps(
  WriteFile* file,
  const DatasetItemWebTimestamps& metadata)
{
  StoreResult result;
  EXP_BACKOFF(
    file->append(sizeof(int),
                 reinterpret_cast<const char*>(&metadata.time_base_numerator)),
    result);
  exit_on_error(result);

  // Width
  EXP_BACKOFF(
    file->append(sizeof(int),
                 reinterpret_cast<const char*>(
                   &metadata.time_base_denominator)),
    result);
  exit_on_error(result);

  size_t num_frames = metadata.pts_timestamps.size();
  EXP_BACKOFF(
    file->append(sizeof(size_t), reinterpret_cast<char*>(&num_frames)),
    result);
  exit_on_error(result);

  EXP_BACKOFF(
    file->append(sizeof(int64_t) * num_frames,
                 reinterpret_cast<const char*>(
                   metadata.pts_timestamps.data())),
    result);
  exit_on_error(result);

  EXP_BACKOFF(
    file->append(sizeof(int64_t) * num_frames,
                 reinterpret_cast<const char*>(
                   metadata.dts_timestamps.data())),
    result);
  exit_on_error(result);
}

DatasetItemWebTimestamps deserialize_dataset_item_web_timestamps(
  RandomReadFile* file,
  uint64_t& pos)
{
  StoreResult result;
  size_t size_read;

  DatasetItemWebTimestamps meta;

  // timebase numerator
  EXP_BACKOFF(
    file->read(pos,
               sizeof(int),
               reinterpret_cast<char*>(&meta.time_base_numerator),
               size_read),
    result);
  exit_on_error(result);
  assert(size_read == sizeof(int));
  pos += size_read;

  // timebase denominator
  EXP_BACKOFF(
    file->read(pos,
               sizeof(int),
               reinterpret_cast<char*>(&meta.time_base_denominator),
               size_read),
    result);
  exit_on_error(result);
  assert(size_read == sizeof(int));
  pos += size_read;

  size_t num_frames;
  // Frames
  EXP_BACKOFF(
    file->read(pos,
               sizeof(size_t),
               reinterpret_cast<char*>(&num_frames),
               size_read),
    result);
  exit_on_error(result);
  assert(size_read == sizeof(size_t));
  pos += size_read;

  meta.pts_timestamps.resize(num_frames);
  meta.dts_timestamps.resize(num_frames);

  EXP_BACKOFF(
    file->read(pos,
               sizeof(int64_t) * num_frames,
               reinterpret_cast<char*>(meta.pts_timestamps.data()),
               size_read),
    result);
  exit_on_error(result);
  assert(size_read == sizeof(int64_t) * num_frames);
  pos += size_read;

  EXP_BACKOFF(
    file->read(pos,
               sizeof(int64_t) * num_frames,
               reinterpret_cast<char*>(meta.dts_timestamps.data()),
               size_read),
    result);
  assert(result == StoreResult::Success ||
         result == StoreResult::EndOfFile);
  assert(size_read == sizeof(int64_t) * num_frames);
  pos += size_read;

  return meta;
}

void serialize_job_descriptor(
  WriteFile* file,
  const JobDescriptor& descriptor)
{
  StoreResult result;

  // Write dataset name
  EXP_BACKOFF(
    file->append(descriptor.dataset_name.size() + 1,
                 descriptor.dataset_name.c_str()),
    result);
  exit_on_error(result);

  // Write out all intervals for each video we processed
  int64_t num_videos = descriptor.intervals.size();
  EXP_BACKOFF(
    file->append(
      sizeof(int64_t),
      (const char*)&num_videos),
    result);
  exit_on_error(result);

  for (auto it : descriptor.intervals) {
    const std::string& video_path = it.first;
    const std::vector<std::tuple<int, int>>& intervals = it.second;

    EXP_BACKOFF(
      file->append(
        video_path.size() + 1,
        video_path.c_str()),
      result);
    exit_on_error(result);

    std::vector<int64_t> buffer;
    int64_t num_intervals = intervals.size();
    buffer.push_back(num_intervals);
    for (const std::tuple<int, int>& interval : intervals) {
      buffer.push_back(std::get<0>(interval));
      buffer.push_back(std::get<1>(interval));
    }
    EXP_BACKOFF(
      file->append(
        buffer.size() * sizeof(int64_t),
        (char*)buffer.data()),
      result);
    exit_on_error(result);
  }
}

JobDescriptor deserialize_job_descriptor(
  RandomReadFile* file,
  uint64_t& file_pos)
{
  JobDescriptor descriptor;

  // Load the entire input
  std::vector<char> bytes = read_entire_file(file, file_pos);;

  char* data = bytes.data();

  descriptor.dataset_name = std::string{data};
  data += descriptor.dataset_name.size() + 1;

  int64_t num_videos = *((int64_t*)data);
  data += sizeof(int64_t);

  for (int64_t i = 0; i < num_videos; ++i) {
    std::string video_path{data};
    data += video_path.size() + 1;

    int64_t num_intervals = *((int64_t*)data);
    data += sizeof(int64_t);
    std::vector<std::tuple<int, int>> intervals;
    for (int64_t j = 0; j < num_intervals; ++j) {
      int64_t start = *((int64_t*)data);
      data += sizeof(int64_t);
      int64_t end = *((int64_t*)data);
      data += sizeof(int64_t);
      intervals.emplace_back(start, end);
    }
    descriptor.intervals[video_path] = intervals;
  }

  return descriptor;
}

}
