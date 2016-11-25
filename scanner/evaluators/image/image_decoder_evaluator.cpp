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

#include "scanner/evaluators/image/image_decoder_evaluator.h"
#include "scanner/evaluators/serialize.h"
#include "scanner/metadata.pb.h"
#include "scanner/util/memory.h"

// For image ingest
#include "jpegwrapper/JPEGReader.h"
#include "lodepng/lodepng.h"

namespace scanner {

ImageDecoderEvaluator::ImageDecoderEvaluator(const EvaluatorConfig& config,
                                             DeviceType device_type)
    : device_type_(device_type), device_id_(config.device_ids[0]) {}

void ImageDecoderEvaluator::configure(const InputFormat& metadata) {
  metadata_ = metadata;
}

void ImageDecoderEvaluator::evaluate(const BatchedColumns& input_columns,
                                     BatchedColumns& output_columns) {
  auto start = now();

  i64 total_frames_decoded = 0;
  i64 total_frames_used = 0;
  size_t num_inputs = input_columns.empty() ? 0 : input_columns[0].rows.size();
  for (size_t i = 0; i < num_inputs; ++i) {
    u8* decode_args_buffer = input_columns[1].rows[i].buffer;
    size_t decode_args_buffer_size = input_columns[1].rows[i].size;

    const u8* in_encoded_buffer = input_columns[0].rows[i].buffer;
    size_t in_encoded_buffer_size = input_columns[0].rows[i].size;

    ImageDecodeArgs args;
    const u8* encoded_buffer;
    size_t encoded_buffer_size = in_encoded_buffer_size;
    if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
      u8* buffer = new u8[decode_args_buffer_size];
      memcpy_buffer(buffer, DeviceType::CPU, 0, decode_args_buffer,
                    DeviceType::GPU, device_id_, decode_args_buffer_size);
      args = deserialize_image_decode_args(buffer, decode_args_buffer_size);
      delete[] buffer;

      buffer = new u8[encoded_buffer_size];
      memcpy_buffer(buffer, DeviceType::CPU, 0, in_encoded_buffer,
                    DeviceType::GPU, device_id_, encoded_buffer_size);
      encoded_buffer = buffer;
#else
      LOG(FATAL) << "Cuda not build.";
#endif
    } else {
      args = deserialize_image_decode_args(decode_args_buffer,
                                           decode_args_buffer_size);
      encoded_buffer = in_encoded_buffer;
    }

    std::vector<i32> valid_frames;
    if (args.sampling() == ImageDecodeArgs::All) {
      const ImageDecodeArgs::StridedInterval& interval = args.interval();
      i32 s = interval.start();
      for (; s < interval.end(); ++s) {
        valid_frames.push_back(s);
      }
    } else if (args.sampling() == ImageDecodeArgs::Strided) {
      const ImageDecodeArgs::StridedInterval& interval = args.interval();
      i32 s = interval.start();
      i32 e = interval.end();
      i32 stride = args.stride();
      for (; s < e; s += stride) {
        valid_frames.push_back(s);
      }
    } else if (args.sampling() == ImageDecodeArgs::Gather) {
      i32 s = 0;
      for (; s < args.gather_points_size(); ++s) {
        valid_frames.push_back(args.gather_points(s));
      }
    } else if (args.sampling() == ImageDecodeArgs::SequenceGather) {
      assert(args.gather_sequences_size() == 1);
      const ImageDecodeArgs::StridedInterval& interval =
          args.gather_sequences(0);
      i32 s = interval.start();
      i32 stride = interval.stride();
      for (; s < interval.end(); s += stride) {
        valid_frames.push_back(s);
      }
    } else {
      assert(false);
    }

    i32 total_output_images = static_cast<i32>(valid_frames.size());

    size_t encoded_buffer_offset = 0;
    i32 current_frame = valid_frames[0];
    i32 current_frame_idx = 0;
    i32 valid_index = 0;
    while (valid_index < total_output_images) {
      auto video_start = now();

      while (current_frame < valid_frames[valid_index]) {
        i32 encoded_image_size = args.compressed_sizes(current_frame_idx);
        encoded_buffer_offset += encoded_image_size;
        current_frame++;
        current_frame_idx++;
      }
      i32 encoded_image_size = args.compressed_sizes(valid_index);
      const u8* encoded_packet = encoded_buffer + encoded_buffer_offset;
      encoded_buffer_offset += encoded_image_size;

      printf("encoded_image size %d, offset %lu\n", encoded_image_size,
             encoded_buffer_offset);

      switch (args.encoding_type()) {
        case ImageEncodingType::JPEG: {
          JPEGReader reader;
          reader.header_mem(const_cast<u8*>(encoded_packet),
                            encoded_image_size);
          if (reader.warnings() != "") {
            LOG(FATAL) << "JPEG file header could not be parsed: "
                       << reader.warnings() << ". Exiting.";
          }
          assert(metadata_.width() == reader.width());
          assert(metadata_.height() == reader.height());
          // switch (reader.colorSpace()) {
          // case JPEG::COLOR_GRAYSCALE:
          //   color_space = ImageColorSpace::Gray;
          //   break;
          // case JPEG::COLOR_RGB:
          // case JPEG::COLOR_YCC:
          // case JPEG::COLOR_CMYK:
          // case JPEG::COLOR_YCCK:
          //   color_space = ImageColorSpace::RGB;
          //   break;
          // case JPEG::COLOR_UNKNOWN:
          //   LOG(FATAL) << "JPEG file " << path << " is of unsupported type: "
          //              << "COLOR_UNKNOWN. Exiting.";
          //   break;
          // }
          i32 frame_size = metadata_.width() * metadata_.height() * 3;
          u8* output = new_buffer(device_type_, device_id_, frame_size);
          std::vector<u8*> rows;
          for (i32 r = 0; r < metadata_.height(); ++r) {
            rows.push_back(output + r * metadata_.width() * 3);
          }
          reader.load(rows.begin());

          output_columns[0].rows.push_back(Row{output, frame_size});
          break;
        }
        case ImageEncodingType::PNG: {
          i32 frame_size = metadata_.width() * metadata_.height() * 3;
          u8* output = new_buffer(device_type_, device_id_, frame_size);
          std::vector<u8*> rows;
          for (i32 r = 0; r < metadata_.height(); ++r) {
            rows.push_back(output + r * metadata_.width() * 3);
          }
          unsigned width;
          unsigned height;
          unsigned error = lodepng_decode24(rows.data(), &width, &height,
                                            encoded_packet, encoded_image_size);
          if (error) {
            LOG(FATAL) << "PNG file could not be parsed: "
                       << lodepng_error_text(error) << ". Exiting.";
          }

          output_columns[0].rows.push_back(Row{output, frame_size});
          break;
        }
        default:
          assert(false);
      }
      valid_index++;
      current_frame++;
      current_frame_idx++;
    }

    if (device_type_ == DeviceType::GPU) {
      delete[] encoded_buffer;
    }
  }

  if (profiler_) {
    profiler_->add_interval("decode", start, now());
    profiler_->increment("effective_frames", total_frames_used);
    profiler_->increment("decoded_frames", total_frames_decoded);
  }
}

ImageDecoderEvaluatorFactory::ImageDecoderEvaluatorFactory(
    DeviceType device_type)
    : device_type_(device_type) {}

EvaluatorCapabilities ImageDecoderEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = device_type_;
  caps.max_devices = 1;
  caps.warmup_size = 0;
  caps.can_overlap = false;
  return caps;
}

std::vector<std::string> ImageDecoderEvaluatorFactory::get_output_names() {
  return {"frame"};
}

Evaluator* ImageDecoderEvaluatorFactory::new_evaluator(
    const EvaluatorConfig& config) {
  return new ImageDecoderEvaluator(config, device_type_);
}
}
