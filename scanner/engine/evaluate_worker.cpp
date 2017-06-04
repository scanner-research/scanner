#include "scanner/engine/evaluate_worker.h"

#include "scanner/engine/op_registry.h"
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
    profiler_(args.profiler) {
}

void PreEvaluateWorker::feed(std::tuple<IOItem, EvalWorkEntry>& entry) {
  auto feed_start = now();

  entry_ = entry;
  IOItem& io_item = std::get<0>(entry);
  EvalWorkEntry& work_entry = std::get<1>(entry);


  needs_configure_ = !(io_item.table_id() == last_table_id_);
  needs_reset_ = true;
  // NOTE(apoms): for avoiding warmup
  // needs_configure_ || !(io_item.item_id() == last_item_id ||
  //       (io_item.table_id() == last_table_id &&
  //        io_item.start_element() == last_end_element));

  last_table_id_ = io_item.table_id();
  last_end_row_ = io_item.end_row();
  last_item_id_ = io_item.item_id();

  // Split up a work entry into work item size chunks
  total_rows_ = - 0;
  for (size_t i = 0; i < work_entry.columns.size(); ++i) {
    total_rows_ =
        std::max(total_rows_, (i64)work_entry.columns[i].size());
  }

  // FIXME: do we need this w/ multiple videos of different resolutions in the same
  // task?
  if (needs_configure_) {
    // decoders_.clear();
  }

  // Setup decoders if they have not been initialized yet
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
      num_devices = num_cpus_;
    }
    for (size_t c = 0; c < work_entry.columns.size(); ++c) {
      if (work_entry.column_types[c] == ColumnType::Video) {
        decoders_.emplace_back(
            new DecoderAutomata(device_handle_, num_devices, decoder_type));
        decoders_.back()->set_profiler(&profiler_);
      }
    }
    profiler_.add_interval("init", init_start, now());
  }

  i32 media_col_idx = 0;
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
      decoders_[media_col_idx]->initialize(args);
      media_col_idx++;
    }
  }
  first_item_ = true;
  current_row_ = 0;
  profiler_.add_interval("feed", feed_start, now());
}

bool PreEvaluateWorker::yield(i32 item_size,
                              std::tuple<IOItem, EvalWorkEntry>& output_entry) {
  if (current_row_ >= total_rows_) return false;

  auto yield_start = now();

  IOItem& io_item = std::get<0>(entry_);
  EvalWorkEntry& work_entry = std::get<1>(entry_);

  i64 r = current_row_;
  current_row_ += item_size;

  bool first_item = (r == 0);
  i32 media_col_idx = 0;
  EvalWorkEntry entry;
  entry.io_item_index = work_entry.io_item_index;
  entry.needs_configure = first_item ? needs_configure_ : false;
  entry.needs_reset = first_item_ ? needs_reset_ : false;
  entry.last_in_task = (r + item_size >= total_rows_) ? true : false;
  entry.warmup_rows = work_entry.warmup_rows;
  entry.columns.resize(work_entry.columns.size());

  i64 start = r;
  i64 end = std::min(r + item_size, total_rows_);
  for (size_t c = 0; c < work_entry.columns.size(); ++c) {
    if (work_entry.column_types[c] == ColumnType::Video) {
      // Perform decoding
      i64 num_rows = end - start;
      if (work_entry.video_encoding_type[media_col_idx] ==
          proto::VideoDescriptor::H264) {
        // Encoded as video
        FrameInfo frame_info(decode_args_[media_col_idx][0].height(),
                             decode_args_[media_col_idx][0].width(), 3,
                             FrameType::U8);
        u8* buffer = new_block_buffer(decoder_output_handle_,
                                      num_rows * frame_info.size(), num_rows);
        decoders_[media_col_idx]->get_frames(buffer, num_rows);
        for (i64 n = 0; n < num_rows; ++n) {
          insert_frame(entry.columns[c],
                       new Frame(frame_info, buffer + frame_info.size() * n));
        }
        entry.column_handles.push_back(decoder_output_handle_);
      } else {
        // Encoded as raw data
        FrameInfo frame_info = work_entry.frame_sizes[media_col_idx];
        for (i64 n = 0; n < num_rows; ++n) {
          Element& e = work_entry.columns[c][start + n];
          assert(e.size == frame_info.size());
          insert_frame(entry.columns[c], new Frame(frame_info, e.buffer));
        }
        entry.column_handles.push_back(work_entry.column_handles[c]);
      }
      media_col_idx++;
    } else {
      entry.columns[c] =
          std::vector<Element>(work_entry.columns[c].begin() + start,
                               work_entry.columns[c].begin() + end);
      entry.column_handles.push_back(work_entry.column_handles[c]);
    }
  }
  entry.row_ids = std::vector<i64>(work_entry.row_ids.begin() + start,
                                   work_entry.row_ids.begin() + end);
  profiler_.add_interval("yield", yield_start, now());

  output_entry = std::make_tuple(io_item, entry);
  return true;
}

EvaluateWorker::EvaluateWorker(const EvaluateWorkerArgs& args)
  : node_id_(args.node_id),
    worker_id_(worker_id_),
    profiler_(args.profiler),
    kernel_factories_(args.kernel_factories),
    live_columns_(args.live_columns),
    dead_columns_(args.dead_columns),
    unused_outputs_(args.unused_outputs),
    column_mapping_(args.column_mapping),
    kernel_stencils_(args.kernel_stencils),
    kernel_batch_sizes_(args.kernel_batch_sizes) {
  for (auto& col : column_mapping_) {
    column_mapping_set_.emplace_back(col.begin(), col.end());
  }
  // Instantiate kernels
  {
    OpRegistry* registry = get_op_registry();
    for (size_t i = 0; i < kernel_factories_.size(); ++i) {
      KernelFactory* factory = std::get<0>(kernel_factories_[i]);
      const KernelConfig& config = std::get<1>(kernel_factories_[i]);
      kernel_devices_.push_back(config.devices[0]);
      kernel_num_outputs_.push_back(registry->get_op_info(factory->get_op_name())
                                       ->output_columns()
                                       .size());

#ifdef HAVE_CUDA
      cudaSetDevice(0);
#endif
      auto kernel = factory->new_instance(config);
      kernel->validate(&args.result);
      VLOG(1) << "Kernel finished validation " << args.result.success();
      if (!args.result.success()) {
        VLOG(1) << "Kernel validate failed: " << args.result.msg();
        THREAD_RETURN_SUCCESS();
      }
      kernels_.emplace_back(kernel);
    }
  }
  assert(kernels_.size() > 0);

  for (auto& kernel : kernels_) {
    kernel->set_profiler(&args.profiler);
  }
  // Setup kernel cache sizes
  stencil_cache_row_ids_.resize(kernel_factories_.size());
  stencil_cache_.resize(kernel_factories_.size());
  stencil_cache_devices_.resize(kernel_factories_.size());
  for (size_t i = 0; i < kernel_factories_.size(); ++i) {
    // Resize stencil cache to be the same size as the number of side output
    // columns at that kernel since we need to save all columns
    stencil_cache_[i].resize(live_columns_[i].size());
  }
  valid_output_rows_.resize(kernel_factories_.size());
  current_valid_idx_.assign(kernel_factories_.size(), 0);
}

void EvaluateWorker::new_task(const std::vector<TaskStream>& task_streams) {
  for (size_t i = 0; i < kernel_factories_.size(); ++i) {
    assert(valid_output_rows_[i].size() == current_valid_idx_[i]);
  }
  valid_output_rows_.clear();
  valid_output_rows_set_.clear();
  current_valid_idx_.clear();
  for (auto& ts : task_streams) {
    valid_output_rows_.push_back(ts.valid_output_rows);
    valid_output_rows_set_.push_back(std::set<i64>(ts.valid_output_rows.begin(),
                                                   ts.valid_output_rows.end()));
    current_valid_idx_.push_back(0);
  }

  // Make the op aware of the format of the data
  for (auto& kernel : kernels_) {
    kernel->reset();
  }

  outputs_yielded_ = 0;
  final_output_handles_.clear();;
  final_output_columns_.clear();
  final_row_ids_.clear();

  // Clear the stencil cache
  for (size_t k = 0; k < kernel_factories_.size(); ++k) {
    std::vector<i32>& kernel_stencil = kernel_stencils_[k];
    bool degenerate_stencil =
        (kernel_stencil.size() == 1 && kernel_stencil[0] == 0);
    std::vector<std::deque<Element>>& kernel_cache = stencil_cache_[k];
    std::vector<DeviceHandle>& kernel_cache_devices = stencil_cache_devices_[k];
    std::deque<i64>& kernel_cache_row_ids = stencil_cache_row_ids_[k];
    auto& row_id_deque = kernel_cache_row_ids;
    while (row_id_deque.size() > 0) {
      assert(!kernel_cache_devices.empty());
      row_id_deque.pop_front();
      for (size_t i = 0; i < kernel_cache.size(); ++i) {
        auto& cache_deque = kernel_cache[i];
        Element element = cache_deque.front();
        // If this kernel has a non-degenerate stencil...
        if (!degenerate_stencil) {
          delete_element(kernel_cache_devices[i], element);
        }
        cache_deque.pop_front();
      }
    }
  }
}

void EvaluateWorker::feed(std::tuple<IOItem, EvalWorkEntry>& entry) {
  entry_ = entry;

  IOItem& io_item = std::get<0>(entry);
  EvalWorkEntry& work_entry = std::get<1>(entry);

  auto feed_start = now();

  current_input_ = 0;
  total_inputs_ = 0;
  for (size_t i = 0; i < work_entry.columns.size(); ++i) {
    total_inputs_ =  // io_item.end_row - io_item.start_row;
        std::max(total_inputs_, (i32)work_entry.columns[i].size());
  }

  std::vector<DeviceHandle> side_output_handles = work_entry.column_handles;
  BatchedColumns side_output_columns = work_entry.columns;
  std::vector<i64> side_row_ids = work_entry.row_ids;

  // For each kernel, produce as much output as can be produced given current
  // input rows and stencil cache.
  for (size_t k = 0; k < kernels_.size(); ++k) {
    const std::string& op_name =
        std::get<0>(kernel_factories_[k])->get_op_name();
    DeviceHandle current_handle = kernel_devices_[k];
    std::unique_ptr<BaseKernel>& kernel = kernels_[k];
    i32 num_output_columns = kernel_num_outputs_[k];
    std::vector<i32>& kernel_stencil = kernel_stencils_[k];
    i32 kernel_batch_size = kernel_batch_sizes_[k];
    std::vector<i64>& kernel_valid_rows = valid_output_rows_[k];
    std::set<i64>& kernel_valid_rows_set = valid_output_rows_set_[k];
    std::vector<std::deque<Element>>& kernel_cache = stencil_cache_[k];
    std::vector<DeviceHandle>& kernel_cache_devices = stencil_cache_devices_[k];
    std::deque<i64>& kernel_cache_row_ids = stencil_cache_row_ids_[k];
    std::vector<i32>& input_column_idx = column_mapping_[k];
    std::set<i32>& input_column_idx_set = column_mapping_set_[k];

    // Move all required values in the side output columns to the proper device
    // for this kernel
    for (i32 i = 0; i < input_column_idx.size(); ++i) {
      i32 in_col_idx = input_column_idx[i];
      assert(in_col_idx < side_output_columns.size());

      // If current op type and input buffer type differ, then move
      // the data in the input buffer into a new buffer which has the same
      // type as the op input
      auto copy_start = now();
      move_if_different_address_space(
          profiler_, side_output_handles[in_col_idx], current_handle,
          side_output_columns[in_col_idx]);
      side_output_handles[in_col_idx] = current_handle;
      profiler_.add_interval("op_marshal", copy_start, now());
    }

    // Copy all side_output_columns into the stencil cache so that we can
    // realign them later when the kernel is able to produce a value
    // at that index
    i64 max_row_id_seen = -1;
    for (i32 i = 0; i < side_row_ids.size(); ++i) {
      kernel_cache_row_ids.push_back(side_row_ids[i]);
      max_row_id_seen = std::max(side_row_ids[i], max_row_id_seen);
    }
    side_row_ids.clear();
    for (i32 i = 0; i < side_output_columns.size(); ++i) {
      i32 col_idx = i;

      // Update stencil cache by taking the reference to the side output columns
      // and clearing them, since we will initialize them with the proper
      // data once we have determined how many rows can be produced
      for (i64 r = 0; r < side_output_columns[i].size(); ++r) {
        kernel_cache[i].push_back(side_output_columns[col_idx][r]);
      }
      if (kernel_cache_devices.empty()) {
        kernel_cache_devices = side_output_handles;
      }
      side_output_columns[i].clear();
    }

    // Determine how many elements can be produced given stencil requirements
    // and the currrent stencil cache extent
    i64 producible_rows = 0;
    for (i64 i = current_valid_idx_[k]; i < kernel_valid_rows.size(); ++i) {
      i64 row = kernel_valid_rows[i];
      if (row + kernel_stencil.back() > max_row_id_seen) {
        break;
      }
      producible_rows++;
    }
    assert(producible_rows > 0);

    // Setup side output columns to reflect the number of valid rows that will
    // be produced from this kernel
    {
      std::vector<ElementList> producible_elements(side_output_columns.size());
      i64 s_idx = 0;
      for (i64 i = 0; i < producible_rows; ++i) {
        // Iterate over all elements in the stencil cache to check if they
        // have the proper id
        i64 valid_row_id = kernel_valid_rows[current_valid_idx_[k] + i];
        for (; s_idx < kernel_cache_row_ids.size(); ++s_idx) {
          if (kernel_cache_row_ids[s_idx] == valid_row_id) {
            side_row_ids.push_back(valid_row_id);
            for (i64 c = 0; c < kernel_cache.size(); ++c) {
              producible_elements[c].push_back(kernel_cache[c][s_idx]);
            }
            s_idx++;
            break;
          }
        }
      }
      assert(producible_elements[0].size() == producible_rows);

      // Update side output data by copying data since we don't have
      // reference counting for all pointers :(
      if (!(kernel_stencil.size() == 1 && kernel_stencil[0] == 0)) {
        for (i64 c = 0; c < side_output_columns.size(); ++c) {
          side_output_columns[c] = duplicate_elements(
              profiler_, side_output_handles[c], side_output_handles[c],
              producible_elements[c]);
        }
      } else {
        // However, if we aren't stenciling, then we don't need to copy since we
        // know it will only be used once
        side_output_columns.swap(producible_elements);
      }
    }

    // NOTE(apoms): the number of producible rows should be a multiple of the
    // batch size. If not, then this should be the last batch in the task.
    // We should add an assert to verify this is the case.
    i64 row_start = current_valid_idx_[k];
    i64 row_end = current_valid_idx_[k] + producible_rows;
    current_valid_idx_[k] += producible_rows;

    for (i32 c = 0; c < num_output_columns; ++c) {
      side_output_handles.push_back(current_handle);
      side_output_columns.emplace_back();
    }
    for (i32 start = row_start; start < row_end; start += kernel_batch_size) {
      i32 batch = std::min((i64)kernel_batch_size, row_end - start);
      i32 end = start + batch;
      // Stage inputs to the kernel using the stencil cache
      StenciledBatchedColumns input_columns(input_column_idx.size());
      // For each column
      auto& cache_row_deque = kernel_cache_row_ids;
      for (size_t i = 0; i < input_column_idx.size(); ++i) {
        i32 col_id = input_column_idx[i];
        auto& cache_deque = kernel_cache[col_id];
        auto& col = input_columns[i];
        col.resize(batch);
        // For each batch element
        for (i64 r = start; r < end; ++r) {
          auto& input_stencil = col[r - start];
          i64 last_cache_element = 0;
          // Place elements in "stencil" dimension of input columns
          i64 curr_row = kernel_valid_rows[r];
          for (i64 s : kernel_stencil) {
            i64 desired_row = curr_row + s;
            // Search for desired stencil element
            for (; last_cache_element < cache_row_deque.size();
                 ++last_cache_element) {
              i64 cache_row_id = cache_row_deque[last_cache_element];
              if (desired_row == cache_row_id) {
                input_stencil.push_back(cache_deque[last_cache_element]);
                break;
              }
            }
          }
          assert(input_stencil.size() == kernel_stencil.size());
        }
      }

      // Setup output buffers to receive op output
      DeviceHandle output_handle = current_handle;
      BatchedColumns output_columns;
      output_columns.resize(num_output_columns);

      // Map from previous output columns to the set of input columns needed
      // by the kernel
      auto eval_start = now();
      kernel->execute_kernel(input_columns, output_columns);
      profiler_.add_interval("evaluate:" + op_name, eval_start, now());
      // Delete unused outputs
      for (size_t y = 0; y < unused_outputs_[k].size(); ++y) {
        i32 unused_col_idx =
            unused_outputs_[k][unused_outputs_[k].size() - 1 - y];
        ElementList& column = output_columns[unused_col_idx];
        for (Element& element : column) {
          delete_element(current_handle, element);
        }
        output_columns.erase(output_columns.begin() + unused_col_idx);
      }

      // Verify the kernel produced the correct amount of output
      for (size_t i = 0; i < output_columns.size(); ++i) {
        LOG_IF(FATAL, output_columns[i].size() != batch)
            << "Op " << k << " produced " << output_columns[i].size()
            << " output elements for column " << i << ". Expected "
            << batch << " outputs.";
      }

      // Add new output columns
      for (size_t cidx = 0; cidx < output_columns.size(); ++cidx) {
        const ElementList& column = output_columns[cidx];
        i32 col_idx =
            side_output_columns.size() - num_output_columns + cidx;
        side_output_columns[col_idx].insert(side_output_columns[col_idx].end(),
                                            column.begin(), column.end());
      }

      // Remove elements from the stencil cache we won't access anymore
      bool degenerate_stencil =
          (kernel_stencil.size() == 1 && kernel_stencil[0] == 0);
      i64 last_cache_element = 0;
      i64 min_used_row =
          kernel_valid_rows[start + batch - kernel_stencil[0]];
      {
        auto& row_id_deque = kernel_cache_row_ids;
        while (row_id_deque.size() > 0) {
          i64 cache_row = row_id_deque.front();
          if (cache_row < min_used_row) {
            row_id_deque.pop_front();
            for (size_t i = 0; i < kernel_cache.size(); ++i) {
              auto device = side_output_handles[i];
              auto& cache_deque = kernel_cache[i];
              Element element = cache_deque.front();
              // If this kernel has a non-degenerate stencil...
              if (!degenerate_stencil) {
                delete_element(device, element);
              }
              cache_deque.pop_front();
            }
          } else {
            break;
          }
        }
      }
    }

    // Delete dead columns
    for (size_t y = 0; y < dead_columns_[k].size(); ++y) {
      i32 dead_col_idx = dead_columns_[k][dead_columns_[k].size() - 1 - y];
      ElementList& column = side_output_columns[dead_col_idx];
      for (Element& element : column) {
        delete_element(side_output_handles[dead_col_idx], element);
      }
      side_output_columns.erase(side_output_columns.begin() + dead_col_idx);
      side_output_handles.erase(side_output_handles.begin() + dead_col_idx);
    }
    // Delete elements from stencil cache that will no longer be used
  }

  final_output_handles_ = side_output_handles;
  if (final_output_columns_.size() == 0) {
    final_output_columns_.resize(side_output_columns.size());
  }
  for (size_t i = 0; i < side_output_columns.size(); ++i) {
    final_output_columns_[i].insert(final_output_columns_[i].end(),
                                    side_output_columns[i].begin(),
                                    side_output_columns[i].end());
  }

  profiler_.add_interval("feed", feed_start, now());
}

bool EvaluateWorker::yield(i32 item_size,
                           std::tuple<IOItem, EvalWorkEntry>& output_entry) {
  IOItem& io_item = std::get<0>(entry_);
  EvalWorkEntry& work_entry = std::get<1>(entry_);

  auto yield_start = now();

  EvalWorkEntry output_work_entry;
  output_work_entry.io_item_index = work_entry.io_item_index;
  output_work_entry.needs_configure = work_entry.needs_configure;
  output_work_entry.needs_reset = work_entry.needs_reset;
  output_work_entry.last_in_task = work_entry.last_in_task;
  output_work_entry.warmup_rows = work_entry.warmup_rows;

  BatchedColumns& work_item_output_columns = output_work_entry.columns;
  std::vector<DeviceHandle>& work_item_output_handles =
      output_work_entry.column_handles;
  i32 num_final_output_columns = 0;

  num_final_output_columns = final_output_columns_.size();
  work_item_output_columns.resize(final_output_columns_.size());
  work_item_output_handles = final_output_handles_;

  i32 yieldable_rows = std::numeric_limits<i32>::max();
  for (i32 i = 0; i < num_final_output_columns; ++i) {
    yieldable_rows =
        std::min((i32)final_output_columns_[i].size(), yieldable_rows);
  }

  for (i32 i = 0; i < num_final_output_columns; ++i) {
    work_item_output_columns[i].insert(
        work_item_output_columns[i].end(), final_output_columns_[i].begin(),
        final_output_columns_[i].begin() + yieldable_rows);
    final_output_columns_[i].erase(
        final_output_columns_[i].begin(),
        final_output_columns_[i].begin() + yieldable_rows);
  }
  output_work_entry.row_ids = std::vector<i64>(
      valid_output_rows_.back().begin() + outputs_yielded_,
      valid_output_rows_.back().begin() + outputs_yielded_ + yieldable_rows);

  assert(output_work_entry.row_ids.size() ==
         work_item_output_columns[0].size());

  outputs_yielded_ += yieldable_rows;

  output_entry = std::make_tuple(io_item, output_work_entry);

  profiler_.add_interval("yield", yield_start, now());

  return true;
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
    if (type != ColumnType::Video || compression_opts.codec == "raw") continue;
    encoders_.emplace_back(
        VideoEncoder::make_from_config(encoder_handle_, 1, encoder_type_));
    encoder_configured_.push_back(false);

    EncodeOptions opts;
    if (compression_opts.codec == "h264") {
      opts.quality = std::atoi(compression_opts.options.at("quality").c_str());
      opts.bitrate = std::atoi(compression_opts.options.at("bitrate").c_str());
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

  current_offset_ = 0;
}

void PostEvaluateWorker::feed(std::tuple<IOItem, EvalWorkEntry>& entry) {
  IOItem& io_item = std::get<0>(entry);
  EvalWorkEntry& work_entry = std::get<1>(entry);

  // Setup row buffer if it was emptied
  if (buffered_entry_.columns.size() == 0) {
    buffered_entry_.io_item_index = work_entry.io_item_index;
    buffered_entry_.columns.resize(column_mapping_.size());
    assert(work_entry.column_handles.size() == columns_.size());
    for (size_t i = 0; i < columns_.size(); ++i) {
      buffered_entry_.column_types.push_back(columns_[i].type());
      buffered_entry_.column_handles.push_back(work_entry.column_handles[i]);
      if (columns_[i].type() == ColumnType::Video) {
        assert(work_entry.columns[i].size() > 0);
        Frame* frame = work_entry.columns[i][0].as_frame();
        buffered_entry_.frame_sizes.push_back(frame->as_frame_info());
      }
      buffered_entry_.compressed.push_back(compression_enabled_[i]);
    }
    if (work_entry.needs_configure) {
      for (size_t i = 0; i < encoder_configured_.size(); ++i) {
        encoder_configured_[i] = false;
      }
    }
  }

  i64 num_rows = work_entry.columns[0].size();
  i32 warmup_frames = work_entry.warmup_rows;
  current_offset_ += num_rows;

  i32 encoder_idx = 0;
  // Swizzle columns correctly
  for (size_t i = 0; i < column_mapping_.size(); ++i) {
    i32 col_idx = column_mapping_[i];
    ColumnType column_type = columns_[i].type();
    // Delete warmup frame outputs
    for (i32 w = 0; w < warmup_frames; ++w) {
      delete_element(work_entry.column_handles[col_idx],
                     work_entry.columns[col_idx][w]);
    }
    // Encode video frames
    if (compression_enabled_[i] && column_type == ColumnType::Video &&
        buffered_entry_.frame_sizes[encoder_idx].type == FrameType::U8) {
      {
        auto start = work_entry.columns[col_idx].begin();
        auto warmup_end = work_entry.columns[col_idx].begin() + warmup_frames;
        work_entry.columns[col_idx].erase(start, warmup_end);
      }
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
      }
      profiler_.add_interval("encode", encode_start, now());
      encoder_idx++;
    } else {
      // Keep non-warmup frame outputs
      buffered_entry_.columns[i].insert(
          buffered_entry_.columns[i].end(),
          work_entry.columns[col_idx].begin() + warmup_frames,
          work_entry.columns[col_idx].end());
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

  encoder_idx = 0;

  // Flush row buffer
  if (work_entry.last_in_task) {
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
          // HACK(apoms): this is really hacky but we put the encoded data in
          // a frame so that we can communicate the frame size downstream
          insert_element(buffered_entry_.columns[i], buffer, actual_size);
        }
        profiler_.add_interval("encode_flush", encode_flush_start, now());
        encoder_configured_[encoder_idx] = false;
        encoder_idx++;
      }
    }

    buffered_entries_.push_back(std::make_tuple(io_item, buffered_entry_));
    buffered_entry_.columns.clear();
  }
}

bool PostEvaluateWorker::yield(std::tuple<IOItem, EvalWorkEntry>& output) {
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
