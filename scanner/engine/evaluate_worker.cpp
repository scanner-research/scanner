#include "scanner/engine/evaluate_worker.h"

#include "scanner/engine/op_registry.h"
#include "scanner/engine/dag_analysis.h"
#include "scanner/util/cuda.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <thread>

namespace scanner {
namespace internal {

PreEvaluateWorker::PreEvaluateWorker(const PreEvaluateWorkerArgs& args)
  : node_id_(args.node_id),
    worker_id_(args.worker_id),
    device_handle_(args.device_handle),
    num_cpus_(args.num_cpus),
    decoder_cpus_(args.decoder_cpus),
    profiler_(args.profiler) {
}

void PreEvaluateWorker::feed(EvalWorkEntry& work_entry, bool first) {
  auto feed_start = now();

  entry_ = work_entry;

  needs_configure_ = !(work_entry.job_index == last_job_idx_);
  needs_reset_ = true;

  last_job_idx_ = work_entry.job_index;

  // Split up a work entry into work item size chunks
  total_rows_ = 0;
  for (size_t i = 0; i < work_entry.columns.size(); ++i) {
    total_rows_ =
        std::max(total_rows_, (i64)work_entry.row_ids[i].size());
  }

  // FIXME: do we need this w/ multiple videos of different resolutions in the
  // same task?
  if (needs_configure_) {
    // decoders_.clear();
  }

  // Setup decoders if they have not been initialized yet
  i32 media_col_idx = 0;
  if (decoders_.empty()) {
    auto init_start = now();
    VideoDecoderType decoder_type;
    i32 num_devices;
    // Select a decoder type based on the type of the first op and
    // the available decoders
    if (device_handle_.type == DeviceType::GPU &&
        VideoDecoder::has_decoder_type(VideoDecoderType::NVIDIA)) {
      decoder_output_handle_.type = DeviceType::GPU;
      decoder_output_handle_.id = device_handle_.id;
      decoder_type = VideoDecoderType::NVIDIA;
      num_devices = 1;
    } else {
      decoder_output_handle_ = CPU_DEVICE;
      decoder_type = VideoDecoderType::SOFTWARE;
      num_devices = decoder_cpus_;
    }
    for (size_t c = 0; c < work_entry.columns.size(); ++c) {
      if (work_entry.column_types[c] == ColumnType::Video &&
          work_entry.video_encoding_type[media_col_idx] ==
              proto::VideoDescriptor::H264) {
        if (work_entry.inplace_video[c]) {
          hwang::DeviceHandle hd;
          switch (device_handle_.type) {
            case DeviceType::CPU:
              hd.type = hwang::DeviceType::CPU;
              break;
            case DeviceType::GPU:
              hd.type = hwang::DeviceType::GPU;
              break;
            default:
              std::abort();
          }
          hd.id = device_handle_.id;

          hwang::VideoDecoderType vd;
          switch (decoder_type) {
            case VideoDecoderType::SOFTWARE:
              vd = hwang::VideoDecoderType::SOFTWARE;
              break;
            case VideoDecoderType::NVIDIA:
              vd = hwang::VideoDecoderType::NVIDIA;
              break;
            default:
              std::abort();
          }

          inplace_decoders_.emplace_back(
              new hwang::DecoderAutomata(hd, num_devices, vd));
          //decoders_.back()->set_profiler(&profiler_);
          decoders_.emplace_back(nullptr);
        } else {
          decoders_.emplace_back(
              new DecoderAutomata(device_handle_, num_devices, decoder_type));
          decoders_.back()->set_profiler(&profiler_);
          inplace_decoders_.emplace_back(nullptr);
        }
        media_col_idx++;
      }
    }
    profiler_.add_interval("init", init_start, now());
  }

  media_col_idx = 0;
  auto setup_start = now();
  // Deserialize all decode args into protobufs
  decode_args_.clear();
  for (size_t c = 0; c < work_entry.columns.size(); ++c) {
    if (work_entry.column_types[c] == ColumnType::Video &&
        work_entry.video_encoding_type[media_col_idx] ==
            proto::VideoDescriptor::H264) {
      decode_args_.emplace_back();
      auto& args = decode_args_.back();
      for (Element element : work_entry.columns[c]) {
        args.emplace_back();
        proto::DecodeArgs& da = args.back();
        google::protobuf::io::ArrayInputStream in_stream(element.buffer,
                                                         element.size);
        google::protobuf::io::CodedInputStream cstream(&in_stream);
        cstream.SetTotalBytesLimit(element.size + 1, element.size + 1);
        bool result = da.ParseFromCodedStream(&cstream);
        assert(result);
        delete_element(CPU_DEVICE, element);
      }

      if (!work_entry.inplace_video[c]) {
        decoders_[media_col_idx]->initialize(args);
      } else {
        // Translate into encoded data
        std::vector<hwang::DecoderAutomata::EncodedData> encoded_data;
        for (auto &da : args) {
          encoded_data.emplace_back();
          hwang::DecoderAutomata::EncodedData& ed = encoded_data.back();
          u8* video_data = reinterpret_cast<u8*>(da.encoded_video());
          size_t video_data_size = da.encoded_video_size();
          ed.encoded_video =
              std::vector<u8>(video_data, video_data + video_data_size);
          ed.width = da.width();
          ed.height = da.height();
          ed.start_keyframe = da.start_keyframe();
          ed.end_keyframe = da.end_keyframe();
          ed.sample_offsets = std::vector<u64>(da.sample_offsets().begin(),
                                               da.sample_offsets().end());
          ed.sample_sizes = std::vector<u64>(da.sample_sizes().begin(),
                                             da.sample_sizes().end());
          ed.keyframes =
              std::vector<u64>(da.keyframes().begin(), da.keyframes().end());
          ed.valid_frames = std::vector<u64>(da.valid_frames().begin(),
                                             da.valid_frames().end());
        }
        if (args.size() > 0) {
          std::vector<u8> metadata(args.back().metadata().begin(),
                                   args.back().metadata().end());
          inplace_decoders_[media_col_idx]->initialize(encoded_data, metadata);
        }
      }
      media_col_idx++;
    }
  }
  first_item_ = first;
  current_row_ = 0;
  profiler_.add_interval("feed", feed_start, now());
}

bool PreEvaluateWorker::yield(i32 item_size,
                              EvalWorkEntry& output_entry) {
  if (current_row_ >= total_rows_) return false;

  auto yield_start = now();

  EvalWorkEntry& work_entry = entry_;

  i64 start_row = current_row_;
  i64 end_row = std::min(current_row_ + item_size, total_rows_);

  bool first_item = (start_row == 0);
  i32 media_col_idx = 0;
  EvalWorkEntry entry;
  entry.table_id = work_entry.table_id;
  entry.job_index = work_entry.job_index;
  entry.task_index = work_entry.task_index;
  entry.needs_configure = first_item ? needs_configure_ : false;
  entry.needs_reset = first_item_ ? needs_reset_ : false;
  entry.last_in_io_packet = (end_row >= total_rows_);
  LOG(INFO) << "end row " << end_row << ", total rows " << total_rows_;
  entry.columns.resize(work_entry.columns.size());
  entry.last_in_task = work_entry.last_in_task;
  entry.row_ids.resize(work_entry.row_ids.size());

  for (size_t c = 0; c < work_entry.columns.size(); ++c) {
    i64 column_start_row =
        std::min(start_row, (i64)work_entry.row_ids.at(c).size());
    i64 column_end_row =
        std::min(end_row, (i64)work_entry.row_ids.at(c).size());
    if (work_entry.column_types[c] == ColumnType::Video) {
      // Perform decoding
      i64 num_rows = column_end_row - column_start_row;
      if (work_entry.video_encoding_type[media_col_idx] ==
          proto::VideoDescriptor::H264) {
        if (num_rows > 0) {
          // Encoded as video
          FrameInfo frame_info(decode_args_[media_col_idx][0].height(),
                               decode_args_[media_col_idx][0].width(), 3,
                               FrameType::U8);
          u8* buffer = new_block_buffer(decoder_output_handle_,
                                        num_rows * frame_info.size(), num_rows);
          if (!work_entry.inplace_video[c]) {
            decoders_[media_col_idx]->get_frames(buffer, num_rows);
          } else {
            inplace_decoders_[media_col_idx]->get_frames(buffer, num_rows);
          }
          for (i64 n = 0; n < num_rows; ++n) {
            insert_frame(entry.columns[c],
                         new Frame(frame_info, buffer + frame_info.size() * n));
          }
        }
        entry.column_handles.push_back(decoder_output_handle_);
      } else {
        // Encoded as raw data
        if (num_rows > 0) {
          FrameInfo frame_info = work_entry.frame_sizes[media_col_idx];
          for (i64 n = 0; n < num_rows; ++n) {
            Element& e = work_entry.columns[c][column_start_row + n];
            assert(e.size == frame_info.size());
            insert_frame(entry.columns[c], new Frame(frame_info, e.buffer));
          }
        }
        entry.column_handles.push_back(work_entry.column_handles[c]);
      }
      media_col_idx++;
    } else {
      entry.columns[c] =
          std::vector<Element>(work_entry.columns[c].begin() + column_start_row,
                               work_entry.columns[c].begin() + column_end_row);
      entry.column_handles.push_back(work_entry.column_handles[c]);
    }
    entry.row_ids[c] =
        std::vector<i64>(work_entry.row_ids[c].begin() + column_start_row,
                         work_entry.row_ids[c].begin() + column_end_row);
  }
  profiler_.add_interval("yield", yield_start, now());

  current_row_ += item_size;

  output_entry = entry;
  return true;
}

EvaluateWorker::EvaluateWorker(const EvaluateWorkerArgs& args)
  : node_id_(args.node_id),
    worker_id_(worker_id_),
    profiler_(args.profiler),
    arg_group_(args.arg_group) {
  auto setup_start = now();
  for (auto& col : arg_group_.column_mapping) {
    column_mapping_set_.emplace_back(col.begin(), col.end());
  }
  // Instantiate kernels
  {
    OpRegistry* registry = get_op_registry();
    DeviceHandle last_device = CPU_DEVICE;
    for (size_t i = 0; i < arg_group_.kernel_factories.size(); ++i) {
      KernelFactory* factory = std::get<0>(arg_group_.kernel_factories[i]);
      if (factory == nullptr) {
        kernel_devices_.push_back(last_device);
        kernel_input_devices_.push_back({last_device});
        kernel_output_devices_.push_back({last_device});
        kernel_num_outputs_.push_back(1);
        kernels_.emplace_back(nullptr);
        continue;
      }
      OpInfo* op_info = registry->get_op_info(factory->get_op_name());
      const KernelConfig& config = std::get<1>(arg_group_.kernel_factories[i]);
      kernel_devices_.push_back(config.devices[0]);
      kernel_input_devices_.emplace_back();
      if (op_info->variadic_inputs()) {
        DeviceHandle handle = config.devices[0];
        for (int i = 0; i < config.input_columns.size(); ++i) {
          kernel_input_devices_.back().push_back(handle);
        }
      } else {
        const auto& input_devices = factory->get_input_devices();
        for (const auto& in_col : op_info->input_columns()) {
          const auto& col_name = in_col.name();
          DeviceType type = config.devices[0].type;
          if (input_devices.count(col_name)) {
            type = input_devices.at(col_name);
          }
          kernel_input_devices_.back().push_back(
              DeviceHandle{type, config.devices[0].id});
        }
      }
      kernel_output_devices_.emplace_back();
      {
        const auto& output_devices = factory->get_output_devices();
        for (const auto& out_col : op_info->output_columns()) {
          const auto& col_name = out_col.name();
          DeviceType type = config.devices[0].type;
          if (output_devices.count(col_name)) {
            type = output_devices.at(col_name);
          }
          kernel_output_devices_.back().push_back(
              DeviceHandle{type, config.devices[0].id});
        }
      }
      last_device = config.devices[0];
      kernel_num_outputs_.push_back(op_info->output_columns().size());

#ifdef HAVE_CUDA
      cudaSetDevice(0);
#endif
      auto kernel = factory->new_instance(config);
      kernel->validate(&args.result);
      VLOG(1) << "Kernel finished validation " << args.result.success();
      if (!args.result.success()) {
        LOG(ERROR) << "Kernel validate failed: " << args.result.msg();
        THREAD_RETURN_SUCCESS();
      }
      kernels_.emplace_back(kernel);
    }
  }
  assert(kernels_.size() > 0);

  for (auto& kernel : kernels_) {
    if (kernel != nullptr) {
      kernel->set_profiler(&args.profiler);
    }
  }
  // Setup kernel cache sizes
  element_cache_row_ids_.resize(kernels_.size());
  element_cache_.resize(kernels_.size());
  element_cache_devices_.resize(kernels_.size());
  for (size_t i = 0; i < kernels_.size(); ++i) {
    // Resize stencil cache to be the same size as the number of inputs
    // to the kernel
    element_cache_[i].resize(arg_group_.column_mapping[i].size());
    element_cache_row_ids_[i].resize(arg_group_.column_mapping[i].size());
  }
  valid_output_rows_.resize(kernels_.size());
  current_valid_input_idx_.resize(kernels_.size());
  current_valid_output_idx_.assign(kernels_.size(), 0);

  args.profiler.add_interval("setup", now(), setup_start);

  // Signal the main worker thread that we've finished startup
  std::unique_lock<std::mutex> lk(args.startup_lock);
  args.startup_count += 1;
  args.startup_cv.notify_one();
}

EvaluateWorker::~EvaluateWorker() {
  // Clear the stencil cache
  clear_stencil_cache();
}

void EvaluateWorker::new_task(i64 job_idx, i64 task_idx,
                              const std::vector<TaskStream>& task_streams) {
  job_idx_ = job_idx;
  task_idx_ = task_idx;
  for (size_t i = 0; i < task_streams.size(); ++i) {
    for (i64 used_rows : current_valid_input_idx_[i]) {
      assert(valid_input_rows_[i].size() == used_rows);
    }
  }
  valid_input_rows_.clear();
  valid_input_rows_set_.clear();
  current_valid_input_idx_.clear();

  compute_rows_.clear();
  compute_rows_set_.clear();
  current_compute_idx_.clear();

  valid_output_rows_.clear();
  valid_output_rows_set_.clear();
  current_valid_output_idx_.clear();

  current_element_cache_input_idx_.clear();
  slice_group_ = -1;
  for (size_t k = 0; k < task_streams.size(); ++k) {
    auto& ts = task_streams[k];
    if (ts.slice_group != -1) {
      slice_group_ = ts.slice_group;
    }
    valid_input_rows_.push_back(ts.valid_input_rows);
    valid_input_rows_set_.push_back(
        std::set<i64>(ts.valid_input_rows.begin(), ts.valid_input_rows.end()));
    current_valid_input_idx_.emplace_back();
    for(i64 i = 0; i < arg_group_.column_mapping[k].size(); ++i) {
      current_valid_input_idx_.back().push_back(0);
    }

    compute_rows_.push_back(ts.compute_input_rows);
    compute_rows_set_.push_back(std::set<i64>(
        ts.compute_input_rows.begin(), ts.compute_input_rows.end()));
    current_compute_idx_.push_back(0);

    valid_output_rows_.push_back(ts.valid_output_rows);
    valid_output_rows_set_.push_back(std::set<i64>(ts.valid_output_rows.begin(),
                                                   ts.valid_output_rows.end()));
    current_valid_output_idx_.push_back(0);

    current_element_cache_input_idx_.push_back(0);
  }

  // Initialize domain samplers for this job and this slice
  domain_samplers_.clear();
  for (auto& kv : arg_group_.sampling_args) {
    i64 op_idx = kv.first;
    i64 slice = 0;
    if (arg_group_.sampling_args.at(op_idx).at(job_idx).size() > 1) {
      slice = slice_group_;
    }
    auto& sampling_args =
        arg_group_.sampling_args.at(op_idx).at(job_idx).at(slice);
    DomainSampler* sampler = nullptr;
    Result result = make_domain_sampler_instance(
        sampling_args.sampling_function(),
        std::vector<u8>(sampling_args.sampling_args().begin(),
                        sampling_args.sampling_args().end()),
        sampler);
    if (!result.success()) {
      VLOG(1) << "Make domain sampler failed: " << result.msg();
      THREAD_RETURN_SUCCESS();
    }
    domain_samplers_[op_idx].reset(sampler);
  }

  // Make the op aware of the format of the data
  for (auto& kernel : kernels_) {
    if (kernel) {
      kernel->reset();
    }
  }

  final_output_handles_.clear();;
  final_output_columns_.clear();
  final_row_ids_.clear();

  clear_stencil_cache();
}

void EvaluateWorker::feed(EvalWorkEntry& work_entry) {
  entry_ = work_entry;

  auto feed_start = now();

  current_input_ = 0;
  total_inputs_ = 0;
  for (size_t i = 0; i < work_entry.columns.size(); ++i) {
    total_inputs_ =  // io_item.end_row - io_item.start_row;
        std::max(total_inputs_, (i32)work_entry.columns[i].size());
  }

  std::vector<DeviceHandle> side_output_handles = work_entry.column_handles;
  BatchedElements side_output_columns = work_entry.columns;
  std::vector<std::vector<i64>> side_row_ids = work_entry.row_ids;

  // For each kernel, produce as much output as can be produced given current
  // input rows and stencil cache.
  for (size_t k = 0; k < arg_group_.op_names.size(); ++k) {
    auto op_start = now();
    const std::string& op_name = arg_group_.op_names.at(k);
    DeviceHandle current_handle = kernel_devices_[k];
    const std::vector<DeviceHandle>& current_input_handles =
        kernel_input_devices_[k];
    const std::vector<DeviceHandle>& current_output_handles =
        kernel_output_devices_[k];

    std::vector<i64>& kernel_valid_input_rows = valid_input_rows_[k];
    std::set<i64>& kernel_valid_input_rows_set = valid_input_rows_set_[k];
    std::vector<i64>& kernel_current_input_idx = current_valid_input_idx_[k];

    std::vector<i64>& kernel_compute_rows = compute_rows_[k];
    i64& kernel_current_compute_idx = current_compute_idx_[k];

    std::vector<i64>& kernel_valid_output_rows = valid_output_rows_[k];
    std::set<i64>& kernel_valid_output_rows_set = valid_output_rows_set_[k];
    i64& kernel_current_output_idx = current_valid_output_idx_[k];

    i64& kernel_element_cache_input_idx = current_element_cache_input_idx_[k];
    std::vector<std::deque<Element>>& kernel_cache = element_cache_[k];
    std::vector<DeviceHandle>& kernel_cache_devices = element_cache_devices_[k];
    std::vector<std::deque<i64>>& kernel_cache_row_ids =
        element_cache_row_ids_[k];
    std::vector<i32>& input_column_idx = arg_group_.column_mapping[k];
    std::set<i32>& input_column_idx_set = column_mapping_set_[k];

    // Since inputs can arrive at different rates, we need to keep
    // inputs around until they have been used.
    // Place all new input elements in side output columns into intermediate
    // cache. If different device, move all required values in the side output
    // columns to the proper device for this kernel
    assert(op_name == INPUT_OP_NAME ||
           current_input_handles.size() == input_column_idx.size());
    if (kernel_cache_devices.empty()) {
      for (i32 i = 0; i < input_column_idx.size(); ++i) {
        kernel_cache_devices.push_back(current_input_handles[i]);
      }
    }
    for (i32 i = 0; i < input_column_idx.size(); ++i) {
      i32 in_col_idx = input_column_idx[i];
      assert(in_col_idx < side_output_columns.size());
      // Select elements which this kernel requires as inputs
      auto& row_ids = side_row_ids[in_col_idx];
      Elements valid_inputs;
      i64& current_input_idx = kernel_current_input_idx[i];
      auto input_create_start = now();
      for (size_t r = 0; r < row_ids.size(); ++r) {
        assert(current_input_idx >= kernel_valid_input_rows.size() ||
               row_ids[r] <= kernel_valid_input_rows[current_input_idx]);
        if (current_input_idx < kernel_valid_input_rows.size() &&
            row_ids[r] == kernel_valid_input_rows[current_input_idx]) {
          // Insert row ids for valid elements into cache
          kernel_cache_row_ids[i].push_back(row_ids[r]);
          Element element(side_output_columns[in_col_idx][r]);
          // We provide the input index to the kernel so that it can detect
          // non-consecutive elements
          element.index = row_ids[r];
          valid_inputs.push_back(element);
          current_input_idx++;
        }
      }
      profiler_.add_interval("input_create", input_create_start, now());
      if (valid_inputs.size() > 0) {
        auto copy_start = now();
        Elements list =
            copy_or_ref_elements(profiler_, side_output_handles[in_col_idx],
                                 current_input_handles[i], valid_inputs);
        profiler_.add_interval("op_marshal", copy_start, now());
        // Insert new elements into cache
        kernel_cache[i].insert(kernel_cache[i].end(), list.begin(), list.end());
      }
    }
    // Determine the highest row seen so we know how many elements we
    // might be able to produce
    i64 max_row_id_seen = -1;
    if (input_column_idx.size() > 0 && kernel_cache_row_ids[0].size() > 0) {
      max_row_id_seen = kernel_cache_row_ids[0].back();
      for (i32 i = 1; i < input_column_idx.size(); ++i) {
        max_row_id_seen =
            std::min(max_row_id_seen, kernel_cache_row_ids[i].back());
      }
      // Update current compute position
      for (i64 i = 0; i < kernel_cache_row_ids[0].size(); ++i) {
        i64 row_id = kernel_cache_row_ids[0][i];
        assert(kernel_current_compute_idx >= kernel_compute_rows.size() ||
               row_id <= kernel_compute_rows[kernel_current_compute_idx]);
        if (kernel_current_compute_idx < kernel_compute_rows.size() &&
            row_id == kernel_compute_rows[kernel_current_compute_idx]) {
          kernel_current_compute_idx++;
        }
      }
    }

    // Figure out how many elements can be produced
    auto compute_producible_elements =
        [kernel_element_cache_input_idx, kernel_current_compute_idx,
         &kernel_compute_rows, max_row_id_seen](i64 stencil, i64 batch) {
          i64 producible_rows = 0;
          for (i64 i = kernel_element_cache_input_idx;
               i < kernel_current_compute_idx; ++i) {
            i64 row = kernel_compute_rows[i];
            // Check if this row was seen by all inputs
            if (row + stencil > max_row_id_seen) {
              break;
            }
            producible_rows++;
          }
          i64 batch_over = producible_rows % batch;
          return producible_rows - batch_over;
        };

    // NOTE(apoms): the number of producible rows should be a multiple of the
    // batch size (if not zero). If not, then this should be the last batch
    // in the task we should add an assert to verify this is the case.
    i64 producible_elements = 0;
    i32 num_output_columns = 0;
    std::vector<i32> kernel_stencil;

    if (is_builtin_op(op_name)) {
      producible_elements = compute_producible_elements(0, 1);
      num_output_columns = 1;
      kernel_stencil = {0};
      if (op_name == INPUT_OP_NAME) {
        num_output_columns = 0;
      }
    } else {
      kernel_stencil = arg_group_.kernel_stencils[k];
      i32 kernel_batch_size = arg_group_.kernel_batch_sizes[k];

      i64 bs = kernel_batch_size;
      // If end of task, we set batch size to 1 to get all remaining elements
      assert(kernel_current_input_idx.size() > 0);
      i64 rows_left_in_task =
          kernel_valid_input_rows.size() - kernel_current_input_idx[0];
      if (rows_left_in_task < kernel_batch_size) {
        bs = 1;
      }
      producible_elements =
          compute_producible_elements(kernel_stencil.back(), bs);

      auto& unused_outputs = arg_group_.unused_outputs[k];
      num_output_columns = kernel_num_outputs_[k] - unused_outputs.size();
    }

    // Grab row ids corresponding to producible elements by walking through
    // element cache
    // NOTE(apoms): elements in kernel cache from each column should be the same
    // since the input domain for all inputs to a kernel must be the same
    std::vector<i64> producible_row_ids(
        kernel_compute_rows.begin() + kernel_element_cache_input_idx,
        kernel_compute_rows.begin() + kernel_element_cache_input_idx +
            producible_elements);

    {
      // Get the output handles for only the columns that are used
      std::vector<DeviceHandle> used_output_column_handles;
      auto& unused_outputs = arg_group_.unused_outputs[k];
      for (i32 c = 0; c < kernel_num_outputs_[k]; ++c) {
        bool found = false;
        for (int i = 0; i < unused_outputs.size(); ++i) {
          if (c == unused_outputs[i]) {
            found = true;
            break;
          }
        }
        if (!found) {
          used_output_column_handles.push_back(current_output_handles[c]);
        }
      }
      for (i32 c = 0; c < num_output_columns; ++c) {
        side_output_handles.push_back(used_output_column_handles[c]);
        side_output_columns.emplace_back();
        side_row_ids.emplace_back();
      }
    }

    auto full_eval_start = now();
    if (op_name == INPUT_OP_NAME) {
      // Should ignore it since we remapped inputs
    } else if (op_name == SAMPLE_OP_NAME) {
      // Filter and remap row ids
      auto& sampler = domain_samplers_.at(k);
      // For each available input, check if it maps to a valid downstream
      // element
      std::vector<i64> downstream_rows;
      std::vector<i64> downstream_upstream_mapping;
      Result result = sampler->get_downstream_rows(
          producible_row_ids, downstream_rows, downstream_upstream_mapping);
      if (!result.success()) {
        VLOG(1) << "Sampler failed: " << result.msg();
        THREAD_RETURN_SUCCESS();
      }

      // Pass down rows and ref elements
      auto& output_column = side_output_columns.back();
      for (size_t i = 0; i < downstream_rows.size(); ++i) {
        i64 upstream_row_idx = downstream_upstream_mapping[i];
        auto& element = *(kernel_cache.at(0).begin() + upstream_row_idx);
        Element ele = add_element_ref(current_handle, element);
        output_column.push_back(ele);
      }
      side_row_ids.back() = downstream_rows;
    } else if (op_name == SPACE_OP_NAME) {
      // Space and remap row ids
      auto& sampler = domain_samplers_.at(k);
      std::vector<i64> downstream_rows;
      std::vector<i64> downstream_upstream_mapping;
      Result result = sampler->get_downstream_rows(
          producible_row_ids, downstream_rows, downstream_upstream_mapping);
      if (!result.success()) {
        VLOG(1) << "Sampler failed: " << result.msg();
        THREAD_RETURN_SUCCESS();
      }
      // For each available input, expand it by placing nulls or repeats
      auto& output_column = side_output_columns.back();
      for (size_t i = 0; i < downstream_rows.size(); ++i) {
        i64 upstream_row_idx = downstream_upstream_mapping[i];
        if (upstream_row_idx == -1) {
          // Put null element
          output_column.emplace_back();
        } else {
          auto& element = *(kernel_cache.at(0).begin() + upstream_row_idx);
          Element ele = add_element_ref(current_handle, element);
          output_column.push_back(ele);
        }
      }
      side_row_ids.back() = downstream_rows;
    } else if (op_name == SLICE_OP_NAME) {
      // Remap row ids from original domain to sub domain
      const auto& slice_output_counts =
          arg_group_.slice_output_rows.at(k).at(job_idx_);
      i64 offset = 0;
      for (i64 i = 0; i < slice_group_; ++i) {
        offset += slice_output_counts.at(i);
      }
      // For each row id, remap it and keep output element the same
      auto& output_column = side_output_columns.back();
      auto& output_row_ids = side_row_ids.back();
      for (size_t i = 0; i < producible_row_ids.size(); ++i) {
        output_row_ids.push_back(producible_row_ids[i] - offset);
        auto& element = *(kernel_cache.at(0).begin() + i);
        Element ele = add_element_ref(current_handle, element);
        output_column.push_back(ele);
      }
    } else if (op_name == UNSLICE_OP_NAME) {
      // Remap row ids from sub domain to original domain
      const auto& unslice_input_counts =
          arg_group_.unslice_input_rows.at(k).at(job_idx_);
      i64 offset = 0;
      for (i64 i = 0; i < slice_group_; ++i) {
        offset += unslice_input_counts.at(i);
      }
      // For each row id, remap it and keep output element the same
      auto& output_column = side_output_columns.back();
      auto& output_row_ids = side_row_ids.back();
      for (size_t i = 0; i < producible_row_ids.size(); ++i) {
        output_row_ids.push_back(producible_row_ids[i] + offset);
        auto& element = *(kernel_cache.at(0).begin() + i);
        Element ele = add_element_ref(current_handle, element);
        output_column.push_back(ele);
      }
    } else {
      assert(!is_builtin_op(op_name));
      // If a regular kernel
      std::unique_ptr<BaseKernel>& kernel = kernels_[k];
      i32 kernel_batch_size = arg_group_.kernel_batch_sizes[k];
      i64 row_start = kernel_element_cache_input_idx;
      i64 row_end = row_start + producible_elements;

      // First build mapping from row number to element in the cache
      auto& cache_row_deque = kernel_cache_row_ids[0];
      std::vector<std::unordered_map<i32, Element>> cache_row_maps;
      for (size_t i = 0; i < input_column_idx.size(); ++i) {
        cache_row_maps.emplace_back();
        auto& row_map = cache_row_maps[i];
        auto& cache_deque = kernel_cache[i];
        for (size_t j = 0; j < cache_row_deque.size(); ++j) {
          row_map[cache_row_deque[j]] = cache_deque[j];
        }
      }

      for (i32 start = row_start; start < row_end; start += kernel_batch_size) {
        i32 batch = std::min((i64)kernel_batch_size, row_end - start);
        i32 end = start + batch;
        // Stage inputs to the kernel using the stencil cache
        StenciledBatchedElements input_columns(input_column_idx.size());
        // For each column
        // NOTE(apoms): choosing the first columns row ids is fine because all
        // input row ids for each column should be the same since all inputs
        // must have the same domain
        auto stencil_create_start = now();
        for (size_t i = 0; i < input_column_idx.size(); ++i) {
          auto& cache_row_map = cache_row_maps[i];
          auto& col = input_columns[i];
          col.resize(batch);
          // For each batch element
          for (i64 r = start; r < end; ++r) {
            auto& input_stencil = col[r - start];
            i64 last_cache_element = 0;
            // Place elements in "stencil" dimension of input columns
            i64 curr_row = kernel_compute_rows[r];
            for (i64 s : kernel_stencil) {
              i64 desired_row = curr_row + s;
              input_stencil.push_back(cache_row_map[desired_row]);
            }
            assert(input_stencil.size() == kernel_stencil.size());
          }
        }
        profiler_.add_interval("stencil_create:" + op_name, stencil_create_start, now());

        // Setup output buffers to receive op output
        BatchedElements output_columns;
        output_columns.resize(num_output_columns);

        // Map from previous output columns to the set of input columns needed
        // by the kernel
        auto eval_start = now();
        kernel->execute_kernel(input_columns, output_columns);
        profiler_.add_interval("evaluate:" + op_name, eval_start, now());

        auto cleanup_start = now();
        // Delete unused output columns
        auto& unused_outputs = arg_group_.unused_outputs[k];
        for (size_t y = 0; y < unused_outputs.size(); ++y) {
          i32 unused_col_idx =
              unused_outputs[unused_outputs.size() - 1 - y];
          Elements& column = output_columns[unused_col_idx];
          for (Element& element : column) {
            delete_element(current_output_handles[unused_col_idx], element);
          }
          output_columns.erase(output_columns.begin() + unused_col_idx);
        }

        // Verify the kernel produced the correct amount of output
        for (size_t i = 0; i < output_columns.size(); ++i) {
          LOG_IF(FATAL, output_columns[i].size() != batch)
              << "Op " << k << " produced " << output_columns[i].size()
              << " output elements for column " << i << ". Expected " << batch
              << " outputs.";
        }

        // Add new output columns
        for (size_t cidx = 0; cidx < output_columns.size(); ++cidx) {
          const Elements& column = output_columns[cidx];
          i32 col_idx = side_output_columns.size() - num_output_columns + cidx;
          side_output_columns[col_idx].insert(
              side_output_columns[col_idx].end(), column.begin(), column.end());
          auto& output_row_ids = side_row_ids[col_idx];
          output_row_ids.insert(
              output_row_ids.end(),
              producible_row_ids.begin() + start - row_start,
              producible_row_ids.begin() + start - row_start + batch);
        }
        profiler_.add_interval("cleanup:" + op_name, cleanup_start, now());
      }
    }
    profiler_.add_interval("full_eval:" + op_name, full_eval_start, now());

    auto full_cleanup_start = now();
    i64 row_start = kernel_element_cache_input_idx;
    i64 row_end = row_start + producible_elements;
    // Filter outputs to only the ones that will be used downstream
    // For each output row, check if it is in the valid output rows
    if (num_output_columns > 0) {
      BatchedElements temp_output_columns(num_output_columns);
      std::vector<std::vector<i64>> temp_row_ids(num_output_columns);

      // For each column, transfer all valid rows to temp output, deleting all
      // the non valid rows, and then swap the temp rows into the side
      // output columns
      i32 first_col_idx = side_output_columns.size() - num_output_columns;
      for (i64 row_start = 0;
           row_start < side_output_columns[first_col_idx].size(); ++row_start) {
        assert(!side_row_ids[first_col_idx].empty());
        // assert(side_row_ids[first_col_idx][row_start] <=
        //        kernel_valid_output_rows[kernel_current_output_idx]);
        if (kernel_current_output_idx < kernel_valid_output_rows.size() &&
            side_row_ids[first_col_idx][row_start] ==
                kernel_valid_output_rows[kernel_current_output_idx]) {
          i64 next_row = kernel_valid_output_rows[kernel_current_output_idx];
          // Is a valid row, so keep
          for (i64 i = 0; i < num_output_columns; ++i) {
            i32 col_idx = side_output_columns.size() - num_output_columns + i;
            auto& element = side_output_columns[col_idx][row_start];
            temp_output_columns[i].push_back(element);
            temp_row_ids[i].push_back(next_row);
          }
          kernel_current_output_idx++;
        } else {
          // Is not a valid row, so delete
          for (i64 i = 0; i < num_output_columns; ++i) {
            i32 col_idx = side_output_columns.size() - num_output_columns + i;
            auto& element = side_output_columns[col_idx][row_start];
            delete_element(side_output_handles[col_idx], element);
          }
        }
      }
      for (i64 i = 0; i < num_output_columns; ++i) {
        i32 col_idx = side_output_columns.size() - num_output_columns + i;
        side_output_columns[col_idx].swap(temp_output_columns[i]);
        side_row_ids[col_idx].swap(temp_row_ids[i]);
      }
    }

    // Remove elements from the element cache we won't access anymore
    if (kernel_valid_input_rows.size() > 0) {
      i64 last_cache_element = 0;
      i64 min_used_row = kernel_valid_input_rows[std::min(
          row_end, (i64)kernel_valid_input_rows.size() - 1)];
      min_used_row += kernel_stencil[0];
      {
        auto& row_id_deque = kernel_cache_row_ids[0];
        while (row_id_deque.size() > 0) {
          i64 cache_row = row_id_deque.front();
          if (cache_row < min_used_row) {
            for (auto& deqs : kernel_cache_row_ids) {
              deqs.pop_front();
            }
            for (size_t i = 0; i < kernel_cache.size(); ++i) {
              auto device = kernel_cache_devices[i];
              auto& cache_deque = kernel_cache[i];
              assert(cache_deque.size() > 0);
              Element element = cache_deque.front();
              delete_element(device, element);
              cache_deque.pop_front();
            }
          } else {
            break;
          }
        }
        kernel_element_cache_input_idx += producible_elements;
      }
    }

    // Remove dead columns from side_output_handles
    // TODO(apoms): move this to before the Op eval
    auto& dead_columns = arg_group_.dead_columns[k];
    for (size_t y = 0; y < dead_columns.size(); ++y) {
      i32 dead_col_idx = dead_columns[dead_columns.size() - 1 - y];
      Elements& column = side_output_columns[dead_col_idx];
      for (Element& element : column) {
        delete_element(side_output_handles[dead_col_idx], element);
      }
      side_output_columns.erase(side_output_columns.begin() + dead_col_idx);
      side_output_handles.erase(side_output_handles.begin() + dead_col_idx);
      side_row_ids.erase(side_row_ids.begin() + dead_col_idx);
    }
    // Delete elements from stencil cache that will no longer be used
    profiler_.add_interval("full_cleanup:" + op_name, full_cleanup_start, now());
    profiler_.add_interval("op:" + op_name, op_start, now());
  }

  final_output_handles_ = side_output_handles;
  if (final_output_columns_.size() == 0) {
    final_output_columns_.resize(side_output_columns.size());
    final_row_ids_.resize(side_output_columns.size());
  }
  for (size_t i = 0; i < side_output_columns.size(); ++i) {
    final_output_columns_[i].insert(final_output_columns_[i].end(),
                                    side_output_columns[i].begin(),
                                    side_output_columns[i].end());
  }
  for (size_t i = 0; i < side_output_columns.size(); ++i) {
    final_row_ids_[i].insert(final_row_ids_[i].end(),
                             side_row_ids[i].begin(),
                             side_row_ids[i].end());
  }

  profiler_.add_interval("feed", feed_start, now());
}

bool EvaluateWorker::yield(i32 item_size, EvalWorkEntry& output_entry) {
  EvalWorkEntry& work_entry = entry_;

  auto yield_start = now();

  EvalWorkEntry output_work_entry;
  output_work_entry.table_id = work_entry.table_id;
  output_work_entry.job_index = work_entry.job_index;
  output_work_entry.task_index = work_entry.task_index;
  output_work_entry.needs_configure = work_entry.needs_configure;
  output_work_entry.needs_reset = work_entry.needs_reset;
  output_work_entry.last_in_io_packet = work_entry.last_in_io_packet;
  output_work_entry.last_in_task = work_entry.last_in_task;

  BatchedElements& work_item_output_columns = output_work_entry.columns;
  std::vector<DeviceHandle>& work_item_output_handles =
      output_work_entry.column_handles;
  std::vector<std::vector<i64>>& work_item_row_ids =
      output_work_entry.row_ids;
  i32 num_final_output_columns = 0;
  num_final_output_columns = final_output_columns_.size();
  work_item_output_columns.resize(num_final_output_columns);
  work_item_output_handles = final_output_handles_;
  work_item_row_ids.resize(num_final_output_columns);

  for (i32 i = 0; i < num_final_output_columns; ++i) {
    work_item_output_columns[i].insert(work_item_output_columns[i].end(),
                                       final_output_columns_[i].begin(),
                                       final_output_columns_[i].end());
    work_item_row_ids[i].insert(work_item_row_ids[i].end(),
                                final_row_ids_[i].begin(),
                                final_row_ids_[i].end());
    final_output_columns_[i].clear();
    final_row_ids_[i].clear();
  }

  output_entry = output_work_entry;

  profiler_.add_interval("yield", yield_start, now());

  return true;
}

void EvaluateWorker::clear_stencil_cache() {
  for (size_t k = 0; k < kernels_.size(); ++k) {
    std::vector<i32>& kernel_stencil = arg_group_.kernel_stencils[k];
    bool degenerate_stencil =
        (kernel_stencil.size() == 1 && kernel_stencil[0] == 0);
    std::vector<std::deque<Element>>& kernel_cache = element_cache_[k];
    std::vector<DeviceHandle>& kernel_cache_devices = element_cache_devices_[k];
    std::vector<std::deque<i64>>& kernel_cache_row_ids =
        element_cache_row_ids_[k];
    auto& input_column_idx = arg_group_.column_mapping[k];
    for (i32 i = 0; i < input_column_idx.size(); ++i) {
      auto& row_id_deque = kernel_cache_row_ids[i];
      row_id_deque.clear();
      auto& cache_deque = kernel_cache[i];
      for (i64 j = 0; j < cache_deque.size(); ++j) {
        assert(!kernel_cache_devices.empty());
        Element element = cache_deque.back();
        delete_element(kernel_cache_devices[i], element);
        cache_deque.pop_back();
      }
    }
  }
}

PostEvaluateWorker::PostEvaluateWorker(const PostEvaluateWorkerArgs& args)
  : profiler_(args.profiler),
    column_mapping_(args.column_mapping),
    columns_(args.columns),
    column_set_(args.column_mapping.begin(), args.column_mapping.end()) {
  assert(args.column_mapping.size() == args.columns.size());

  encoder_handle_ = CPU_DEVICE;
  encoder_type_ = VideoEncoderType::SOFTWARE;

  // Setup video encoders
  // TODO(apoms): Make this dynamic based on the encoded column type
  for (size_t i = 0; i < args.columns.size(); ++i) {
    auto& col = args.columns[i];
    auto& compression_opts = args.column_compression[i];
    ColumnType type = col.type();
    if (type != ColumnType::Video) continue;

    frame_size_initialized_.push_back(false);

    if (compression_opts.codec == "raw") continue;
    encoders_.emplace_back(
        VideoEncoder::make_from_config(encoder_handle_, 1, encoder_type_));
    encoder_configured_.push_back(false);

    EncodeOptions opts;
    if (compression_opts.codec == "h264") {
      opts.quality = std::atoi(compression_opts.options.at("quality").c_str());
      opts.bitrate = std::atoi(compression_opts.options.at("bitrate").c_str());
      opts.keyframe_distance =
          std::atoi(compression_opts.options.at("keyframe_distance").c_str());
    }
    encode_options_.push_back(opts);
  }
  for (auto& compression_opts : args.column_compression) {
    auto& codec = compression_opts.codec;
    bool enabled = true;
    if (codec == "raw") {
      enabled = false;
    }
    compression_enabled_.push_back(enabled);
  }
}

void PostEvaluateWorker::feed(EvalWorkEntry& entry) {
  EvalWorkEntry& work_entry = entry;

  // Setup row buffer if it was emptied
  {
    i32 encoder_idx = 0;
    if (buffered_entry_.columns.size() == 0) {
      buffered_entry_.table_id = work_entry.table_id;
      buffered_entry_.job_index = work_entry.job_index;
      buffered_entry_.task_index = work_entry.task_index;
      buffered_entry_.last_in_task = work_entry.last_in_task;
      buffered_entry_.columns.resize(column_mapping_.size());
      buffered_entry_.row_ids.resize(column_mapping_.size());
      assert(work_entry.column_handles.size() == columns_.size());
      buffered_entry_.column_types.clear();
      buffered_entry_.column_handles.clear();
      buffered_entry_.frame_sizes.clear();
      buffered_entry_.compressed.clear();
      for (size_t i = 0; i < columns_.size(); ++i) {
        i32 col_idx = column_mapping_[i];
        buffered_entry_.column_types.push_back(columns_[i].type());
        buffered_entry_.column_handles.push_back(CPU_DEVICE);
        buffered_entry_.compressed.push_back(compression_enabled_[i]);
        if (columns_[i].type() == ColumnType::Video) {
          buffered_entry_.frame_sizes.emplace_back();
          frame_size_initialized_[encoder_idx] = false;
          encoder_idx++;
        }
      }
      if (work_entry.needs_configure) {
        for (size_t i = 0; i < encoder_configured_.size(); ++i) {
          encoder_configured_[i] = false;
        }
      }
    }
  }

  // Swizzle columns correctly
  {
    i32 encoder_idx = 0;
    for (size_t i = 0; i < column_mapping_.size(); ++i) {
      i32 col_idx = column_mapping_[i];
      ColumnType column_type = columns_[i].type();

      if (work_entry.columns[col_idx].empty()) {
        // No frames yet, so skip this column
        if (column_type == ColumnType::Video) {
          encoder_idx++;
        }
        continue;
      }

      // Initialize frame size if not done so yet
      if (column_type == ColumnType::Video &&
          !frame_size_initialized_[encoder_idx]) {
        assert(work_entry.columns[col_idx].size() > 0);
        Frame* frame = work_entry.columns[col_idx][0].as_frame();
        buffered_entry_.frame_sizes[encoder_idx] = frame->as_frame_info();
        frame_size_initialized_[encoder_idx] = true;
      }

      // Encode video frames
      if (compression_enabled_[i] && column_type == ColumnType::Video &&
          buffered_entry_.frame_sizes[encoder_idx].type == FrameType::U8) {
        auto& encoder = encoders_[encoder_idx];
        if (!encoder_configured_[encoder_idx]) {
          // Configure encoder
          encoder_configured_[encoder_idx] = true;
          Frame* frame = work_entry.columns[col_idx][0].as_frame();
          encoder->configure(frame->as_frame_info(),

                             encode_options_[encoder_idx]);
        }

        // Move frames to device for the encoder
        move_if_different_address_space(
            profiler_, work_entry.column_handles[col_idx], encoder_handle_,
            work_entry.columns[col_idx]);

        // Pass frames into encoder
        auto encode_start = now();
        for (auto& row : work_entry.columns[col_idx]) {
          Frame* frame = row.as_frame();
          bool new_packet = encoder->feed(frame->data, frame->size());
          while (new_packet) {
            size_t buffer_size = 4 * 1024 * 1024;
            u8* buffer = new_buffer(CPU_DEVICE, buffer_size);
            size_t actual_size;
            new_packet = encoder->get_packet(buffer, buffer_size, actual_size);
            LOG_IF(FATAL, new_packet && actual_size > buffer_size)
                << "Packet buffer not large enough (" << buffer_size << " vs "
                << actual_size << ")";
            insert_element(buffered_entry_.columns[i], buffer, actual_size);
          }
          delete_element(encoder_handle_, row);
        }
        profiler_.add_interval("encode", encode_start, now());
        encoder_idx++;
      } else {
        // Move data to CPU to avoid overflow on GPU
        move_if_different_address_space(
            profiler_, work_entry.column_handles[col_idx], CPU_DEVICE,
            work_entry.columns[col_idx]);
        buffered_entry_.columns[i].insert(buffered_entry_.columns[i].end(),
                                          work_entry.columns[col_idx].begin(),
                                          work_entry.columns[col_idx].end());
        buffered_entry_.row_ids[i].insert(buffered_entry_.row_ids[i].end(),
                                          work_entry.row_ids[col_idx].begin(),
                                          work_entry.row_ids[col_idx].end());
      }
    }

    // Delete unused columns
    for (size_t i = 0; i < work_entry.columns.size(); ++i) {
      if (column_set_.count(i) > 0) {
        continue;
      }
      for (i32 b = 0; b < work_entry.columns[i].size(); ++b) {
        delete_element(work_entry.column_handles[i], work_entry.columns[i][b]);
      }
    }
  }

  // Flush row buffer
  if (work_entry.last_in_io_packet) {
    i32 encoder_idx = 0;
    // Flush video encoder and get rest of packets
    for (size_t i = 0; i < column_mapping_.size(); ++i) {
      ColumnType column_type = columns_[i].type();
      if (compression_enabled_[i] && column_type == ColumnType::Video &&
          buffered_entry_.frame_sizes[encoder_idx].type == FrameType::U8) {
        auto& encoder = encoders_[encoder_idx];

        // Get last packets in encoder
        auto encode_flush_start = now();
        bool new_packet = encoder->flush();
        while (new_packet) {
          size_t buffer_size = 4 * 1024 * 1024;
          u8* buffer = new_buffer(CPU_DEVICE, buffer_size);
          size_t actual_size;
          new_packet = encoder->get_packet(buffer, buffer_size, actual_size);
          LOG_IF(FATAL, new_packet && actual_size > buffer_size)
              << "Packet buffer not large enough (" << buffer_size << " vs "
              << actual_size << ")";
          insert_element(buffered_entry_.columns[i], buffer, actual_size);
        }
        profiler_.add_interval("encode_flush", encode_flush_start, now());
        encoder_configured_[encoder_idx] = false;
        encoder_idx++;
      }
    }

    assert(buffered_entry_.columns.size() > 0 &&
           buffered_entry_.columns[0].size() > 0);
    // Only push an entry if it is non empty
    if (buffered_entry_.columns.size() > 0 &&
        buffered_entry_.columns[0].size() > 0) {
      buffered_entries_.push_back(buffered_entry_);
      buffered_entry_.columns.clear();
      buffered_entry_.row_ids.clear();
    }
  }
}

bool PostEvaluateWorker::yield(EvalWorkEntry& output) {
  auto yield_start = now();

  bool got_result = false;
  if (buffered_entries_.size() > 0) {
    output = buffered_entries_.front();
    buffered_entries_.pop_front();
    got_result = true;
  }

  profiler_.add_interval("yield", yield_start, now());
  return got_result;
}

}
}
