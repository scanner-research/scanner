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
#include "bitmap-cpp/bitmap.h"

#ifdef HAVE_CUDA
#include "scanner/util/cuda.h"
#endif

namespace scanner {

ImageDecoderEvaluator::ImageDecoderEvaluator(const EvaluatorConfig& config,
                                             DeviceType device_type)
    : device_type_(device_type), device_id_(config.device_ids[0]) {}

void ImageDecoderEvaluator::configure(const BatchConfig& config) {
  config_ = config;
  assert(config.formats.size() == 1);
  frame_width_ = config.formats[0].width();
  frame_height_ = config.formats[0].height();
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
      memcpy_buffer(buffer, CPU_DEVICE, decode_args_buffer,
                    {DeviceType::GPU, device_id_}, decode_args_buffer_size);
      args = deserialize_image_decode_args(buffer, decode_args_buffer_size);
      delete[] buffer;

      buffer = new u8[encoded_buffer_size];
      memcpy_buffer(buffer, CPU_DEVICE, in_encoded_buffer,
                    {DeviceType::GPU, device_id_}, encoded_buffer_size);
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

      i32 encoded_image_size = args.compressed_sizes(current_frame_idx);
      const u8* encoded_packet = encoded_buffer + encoded_buffer_offset;
      encoded_buffer_offset += encoded_image_size;

      // printf("encoded_image size %d, offset %lu\n", encoded_image_size,
      //        encoded_buffer_offset);

      i32 frame_size = frame_width_ * frame_height_ * 3;
      u8* output = new_buffer({device_type_, device_id_}, frame_size);
      switch (args.encoding_type()) {
        case ImageEncodingType::JPEG: {
          try {
            JPEGReader reader;
            reader.header_mem(const_cast<u8*>(encoded_packet),
                              encoded_image_size);
            if (reader.warnings() != "") {
              LOG(FATAL) << "JPEG file header could not be parsed: "
                         << reader.warnings() << ". Exiting.";
            }
            assert(frame_width_ == reader.width());
            assert(frame_height_ == reader.height());
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
            //   LOG(FATAL) << "JPEG file " << path << " is of unsupported type:
            //   "
            //              << "COLOR_UNKNOWN. Exiting.";
            //   break;
            // }
            std::vector<u8*> rows;
            for (i32 r = 0; r < metadata_.height(); ++r) {
              rows.push_back(output + r * frame_width_ * 3);
            }
            reader.load(rows.begin());
          } catch (const std::exception& e) {
            LOG(FATAL) << "Failed to load JPEG with error: " << e.what();
          }
          break;
        }
        case ImageEncodingType::PNG: {
          std::vector<u8*> rows;
          for (i32 r = 0; r < frame_height_; ++r) {
            rows.push_back(output + r * frame_width_ * 3);
          }
          unsigned width;
          unsigned height;
          unsigned error = lodepng_decode24(rows.data(), &width, &height,
                                            encoded_packet, encoded_image_size);
          if (error) {
            LOG(FATAL) << "PNG file could not be parsed: "
                       << lodepng_error_text(error) << ". Exiting.";
          }

          break;
        }
      case ImageEncodingType::BMP: {
        bitmap::DecodeResult result =
          bitmap::bitmap_decode(encoded_packet, encoded_image_size, output);
        LOG_IF(FATAL, result != bitmap::DecodeResult::Success)
          << "BMP file could not be parsed";
        break;
      }
      default:
        assert(false);
      }
      output_columns[0].rows.push_back(Row{output, frame_size});
      valid_index++;
      current_frame_idx++;
      current_frame++;
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

std::vector<std::string> ImageDecoderEvaluatorFactory::get_output_columns(
    const std::vector<std::string>& input_columns) {
  return {"frame"};
}

Evaluator* ImageDecoderEvaluatorFactory::new_evaluator(
    const EvaluatorConfig& config) {
  return new ImageDecoderEvaluator(config, device_type_);
}
}
