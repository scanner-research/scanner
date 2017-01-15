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

#include "comparison/peak/peak_video_decoder.h"
#include "scanner/video/software/software_video_decoder.h"

#include "scanner/util/queue.h"
#include "scanner/util/util.h"
#include "scanner/util/h264.h"
#include "scanner/engine/halide_context.h"

#include "scanner/evaluators/caffe/net_descriptor.h"
#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"

#include "caffe_input_transformer_gpu/caffe_input_transformer_gpu.h"
#include "HalideRuntimeCuda.h"

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/errors.hpp>

#ifdef HAVE_CUDA
#include "scanner/util/cuda.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/error.h"
#include "libavutil/frame.h"
#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
#include "libswscale/swscale.h"
}

#include <iostream>
#include <fstream>
#include <thread>
#include <unistd.h>
#include <mutex>

using namespace scanner;

namespace po = boost::program_options;
namespace cvc = cv::cuda;

using scanner::NetDescriptor;
using scanner::Queue;


enum OpType {
  Histogram,
  Flow,
  Caffe
};

struct BufferHandle {
  u8* buffer;
  int elements;
};

struct SaveHandle {
  u8* buffer;
  cudaStream_t stream;
  int elements;
};

using DecoderFn = std::function<void(int, Queue<std::string>&, Queue<u8*> &,
                                     Queue<BufferHandle> &)>;

using WorkerFn = std::function<void(int, Queue<u8 *> &, Queue<BufferHandle> &,
                                    Queue<SaveHandle> &, Queue<SaveHandle> &)>;

const int NUM_BUFFERS = 5;
int BATCH_SIZE = 192;      // Batch size for network
const int NET_BATCH_SIZE = 96;      // Batch size for network
const int FLOW_WORK_REDUCTION = 20;
const int BINS = 16;
const std::string NET_PATH = "features/googlenet.toml";

int GPUS_PER_NODE = 1;           // GPUs to use per node
int width;
int height;
size_t output_element_size;
size_t output_buffer_size;

std::string PATH;
std::string OPERATION;
std::mutex TIMINGS_MUTEX;
std::map<std::string, double> TIMINGS;

bool IS_CPU;

std::string output_path() {
  i32 idx = 0;
  return "/tmp/peak_outputs/videos" + std::to_string(idx) + ".bin";
}

struct CodecState {
  AVPacket packet;
  AVFrame* frame;
  AVFormatContext* format_context;
  AVCodec* in_codec;
  AVCodecContext* cc;
  SwsContext* sws_context;
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 34, 0)
  AVCodecParameters* in_cc_params;
#endif
  i32 video_stream_index;
  AVBitStreamFilterContext* annexb;
};

bool setup_video_codec(CodecState& state, const std::string& path) {
  LOG(INFO) << "Setting up video codec";
  av_init_packet(&state.packet);
  state.frame = av_frame_alloc();
  state.format_context = avformat_alloc_context();

  avformat_open_input(&state.format_context, path.c_str(), 0, 0);

  // Read file header
  LOG(INFO) << "Opening input file to read format";
  if (avformat_open_input(&state.format_context, NULL, NULL, NULL) < 0) {
    LOG(ERROR) << "open input failed";
    return false;
  }
  // Some formats don't have a header
  LOG(INFO) << "Find stream info";
  if (avformat_find_stream_info(state.format_context, NULL) < 0) {
    LOG(ERROR) << "find stream info failed";
    return false;
  }

  LOG(INFO) << "Dimp format";
  av_dump_format(state.format_context, 0, NULL, 0);

  // Find the best video stream in our input video
  LOG(INFO) << "Find best stream";
  state.video_stream_index = av_find_best_stream(
      state.format_context, AVMEDIA_TYPE_VIDEO, -1 /* auto select */,
      -1 /* no related stream */, &state.in_codec, 0 /* flags */);
  if (state.video_stream_index < 0) {
    LOG(ERROR) << "could not find best stream";
    return false;
  }

  AVStream const* const in_stream =
      state.format_context->streams[state.video_stream_index];

  LOG(INFO) << "Find decoder";
  state.in_codec = avcodec_find_decoder(AV_CODEC_ID_H264);
  if (state.in_codec == NULL) {
    LOG(FATAL) << "could not find h264 decoder";
  }

  state.cc = avcodec_alloc_context3(state.in_codec);
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 34, 0)
  state.in_cc_params = avcodec_parameters_alloc();
  if (avcodec_parameters_from_context(state.in_cc_params, in_stream->codec) <
      0) {
    LOG(ERROR) << "could not copy codec params from input stream";
    return false;
  }
  if (avcodec_parameters_to_context(state.cc, state.in_cc_params) < 0) {
    LOG(ERROR) << "could not copy codec params to in cc";
    return false;
  }
#else
  if (avcodec_copy_context(state.cc, in_stream->codec) < 0) {
    LOG(ERROR) << "could not copy codec params to in cc";
    return false;
  }
#endif

  state.cc->thread_count = 16;
  state.cc->refcounted_frames = 1;
  if (avcodec_open2(state.cc, state.in_codec, NULL) < 0) {
    LOG(ERROR) << "could not open codec";
    return false;
  }

//state.annexb = av_bitstream_filter_init("h264_mp4toannexb");

  return true;
}

void cleanup_video_codec(CodecState state) {
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(55, 53, 0)
  avcodec_free_context(&state.cc);
#else
  avcodec_close(state.cc);
  av_freep(&state.cc);
#endif
  avformat_close_input(&state.format_context);
  av_frame_free(&state.frame);
  //av_bitstream_filter_close(state.annexb);
}

void cpu_decoder_worker(int gpu_device_id, Queue<std::string> &video_paths,
                        Queue<u8 *> &free_buffers,
                        Queue<BufferHandle> &decoded_frames) {
  double decode_time = 0;

  int64_t frame_size = width * height * 4 * sizeof(u8);
  int64_t frame = 0;
  while (true) {
    std::string path;
    video_paths.pop(path);

    if (path == "") {
      break;
    }

    printf("popped %s\n", path.c_str());
    CodecState state;
    bool success = setup_video_codec(state, path);
    LOG_IF(FATAL, !success) << "Did not setup video codec";

    int64_t frame = 0;
    bool video_done = false;
    while (!video_done) {
      u8 *buffer;
      free_buffers.pop(buffer);

      int64_t buffer_frame = 0;
      while (buffer_frame < BATCH_SIZE) {
        auto decode_start = scanner::now();
        int error = av_read_frame(state.format_context, &state.packet);
        if (error == AVERROR_EOF) {
          video_done = true;
          break;
        }
        if (state.packet.stream_index != state.video_stream_index) {
          av_packet_unref(&state.packet);
          continue;
        }

        error = avcodec_send_packet(state.cc, &state.packet);
        if (error != AVERROR_EOF) {
          if (error < 0) {
            char err_msg[256];
            av_strerror(error, err_msg, 256);
            fprintf(stderr, "Error while sending packet (%d): %s\n", error,
                    err_msg);
            LOG(FATAL) << "Error while sending packet";
          }
        }
        av_packet_unref(&state.packet);
        while (buffer_frame < BATCH_SIZE) {
          error = avcodec_receive_frame(state.cc, state.frame);
          if (error == AVERROR_EOF) {
            av_frame_unref(state.frame);
            break;
          }
          if (error == 0) {
            u8* scale_buffer = buffer + buffer_frame * frame_size;

            uint8_t *out_slices[4];
            int out_linesizes[4];
            int required_size = av_image_fill_arrays(
                out_slices, out_linesizes, scale_buffer, AV_PIX_FMT_RGB24,
                width, height, 1);

            AVPixelFormat decoder_pixel_format = state.cc->pix_fmt;
            if (state.sws_context == nullptr) {
              printf("sws %d\n", decoder_pixel_format);
              state.sws_context = sws_getContext(
                  width, height, decoder_pixel_format, width,
                  height, AV_PIX_FMT_RGB24, SWS_BICUBIC, NULL, NULL, NULL);
            }
            if (state.sws_context == NULL) {
              LOG(FATAL) << "Could not get sws context";
            }

            av_frame_unref(state.frame);
            if (sws_scale(state.sws_context, state.frame->data,
                          state.frame->linesize, 0, state.frame->height,
                          out_slices, out_linesizes) < 0) {
              LOG(FATAL) << "Sws_scale failed";
              exit(EXIT_FAILURE);
            }

            av_frame_unref(state.frame);
            frame++;
            buffer_frame++;
            continue;
          } else if (error == AVERROR(EAGAIN)) {
            break;
          } else {
            char err_msg[256];
            av_strerror(error, err_msg, 256);
            fprintf(stderr, "Error while receiving frame (%d): %s\n", error,
                    err_msg);
            exit(-1);
          }
        }
      }

      printf("video %s, %d\n", path.c_str(), frame);
      BufferHandle h;
      h.buffer = buffer;
      h.elements = buffer_frame;
      decoded_frames.push(h);
    }
    cleanup_video_codec(state);
  }
  std::unique_lock<std::mutex> lock(TIMINGS_MUTEX);
  TIMINGS["decode"] += decode_time;
}



void decoder_worker(int gpu_device_id,
                    Queue<std::string> &video_paths,
                    Queue<u8 *> &free_buffers,
                    Queue<BufferHandle> &decoded_frames) {
  double decode_time = 0;

  int64_t frame_size = width * height * 4 * sizeof(u8);
  int64_t frame = 0;
  std::vector<u8> frame_buffer(frame_size);

  cv::VideoCapture cap;
  cv::Ptr<cv::cudacodec::VideoReader> video;
  while (true) {
    std::string path;
    video_paths.pop(path);

    if (path == "") {
      break;
    }

    printf("popped %s\n", path.c_str());
    if (IS_CPU) {
      cap.open(path);
    } else {
      video = cv::cudacodec::createVideoReader(path);
    }

    int64_t frame = 0;
    bool video_done = false;
    while (!video_done) {
      u8 *buffer;
      free_buffers.pop(buffer);

      int64_t buffer_frame = 0;
      while (buffer_frame < BATCH_SIZE) {
        bool valid;
        auto decode_start = scanner::now();
        if (IS_CPU) {
          cv::Mat cpu_image(width, height, CV_8UC3,
                            buffer + buffer_frame * frame_size);
          valid = cap.read(cpu_image);
        } else {
          cvc::GpuMat gpu_image(width, height, CV_8UC4,
                                buffer + buffer_frame * frame_size);
          valid = video->nextFrame(gpu_image);
        }
        decode_time += scanner::nano_since(decode_start);
        if (!valid) {
          video_done = true;
          break;
        }
        buffer_frame++;
        frame++;
      }

      printf("video %s, %d\n", path.c_str(), frame);
      BufferHandle h;
      h.buffer = buffer;
      h.elements = buffer_frame;
      decoded_frames.push(h);
    }
  }
  std::unique_lock<std::mutex> lock(TIMINGS_MUTEX);
  TIMINGS["decode"] += decode_time;
}

void save_worker(int gpu_device_id, Queue<SaveHandle> &save_buffers,
                 Queue<SaveHandle> &free_output_buffers) {
  int64_t frame_size = width * height * 3 * sizeof(u8);
  int64_t frame = 0;
  double save_time = 0;

  u8* buf;
  if (!IS_CPU) {
    cudaMallocHost(&buf, output_buffer_size);
  }

  std::ofstream outfile(output_path(),
                        std::fstream::binary | std::fstream::trunc);
  assert(outfile.good());
  while (true) {
    SaveHandle handle;
    save_buffers.pop(handle);

    if (handle.buffer == nullptr) {
      break;
    }

    // Copy data down
    auto save_start = scanner::now();
    if (IS_CPU) {
      buf = handle.buffer;
    } else {
      cudaMemcpyAsync(buf, handle.buffer, output_element_size * handle.elements,
                      cudaMemcpyDefault, handle.stream);
      // Sync so we know it is done
      cudaStreamSynchronize(handle.stream);
    }
    // Write out
    if (OPERATION != "flow") {
      outfile.write((char*)buf, output_element_size * handle.elements);
    }
    save_time += scanner::nano_since(save_start);

    free_output_buffers.push(handle);
  }
  auto save_start = scanner::now();
  outfile.close();
  save_time += scanner::nano_since(save_start);
  std::unique_lock<std::mutex> lock(TIMINGS_MUTEX);
  TIMINGS["save"] += save_time;
  //cudaFreeHost(buf);
}

void video_histogram_cpu_worker(int gpu_device_id,
                                Queue<u8 *> &free_buffers,
                                Queue<BufferHandle> &decoded_frames,
                                Queue<SaveHandle> &free_output_buffers,
                                Queue<SaveHandle> &save_buffers) {
  double setup_time = 0;
  double load_time = 0;
  double histo_time = 0;
  double save_time = 0;

  // Set ourselves to the correct GPU
  int frame_size = width * height * 4 * sizeof(u8);

  auto setup_start = scanner::now();
  std::vector<cv::Mat> planes;
  for (int i = 0; i < 3; ++i) {
    planes.push_back(cv::Mat(height, width, CV_8UC1));
  }

  cv::Mat hist;
  cv::Mat hist_32s;

  int64_t frame = 0;
  while (true) {
    BufferHandle buffer_handle;
    decoded_frames.pop(buffer_handle);
    u8* buffer = buffer_handle.buffer;
    int elements = buffer_handle.elements;
    if (buffer == nullptr) {
      break;
    }

    SaveHandle save_handle;
    free_output_buffers.pop(save_handle);
    u8* output_buffer = save_handle.buffer;
    save_handle.elements = elements;

    for (int i = 0; i < elements; ++i) {
      cv::Mat out(1, BINS * 3, CV_32S, output_buffer + i * output_element_size);
      cv::Mat image(height, width, CV_8UC3, buffer + i * frame_size);
      auto histo_start = scanner::now();
      float range[] = {0, 256};
      const float* histRange = {range};
      u8* output_buf = output_buffer + i * output_element_size;
      for (i32 j = 0; j < 3; ++j) {
        int channels[] = {j};
        cv::Mat out(BINS, 1, CV_32S, output_buf + BINS * sizeof(int));
        cv::calcHist(&image, 1, channels, cv::Mat(),
                     out,
                     1, &BINS,
                     &histRange);
      }
      histo_time += scanner::nano_since(histo_start);
    }

    save_buffers.push(save_handle);
    free_buffers.push(buffer);
  }
  std::unique_lock<std::mutex> lock(TIMINGS_MUTEX);
  TIMINGS["setup"] += setup_time;
  TIMINGS["load"] += load_time;
  TIMINGS["eval"] += histo_time;
  TIMINGS["save"] += save_time;
}

void video_histogram_gpu_worker(int gpu_device_id, Queue<u8 *> &free_buffers,
                                Queue<BufferHandle> &decoded_frames,
                                Queue<SaveHandle> &free_output_buffers,
                                Queue<SaveHandle> &save_buffers) {
  double setup_time = 0;
  double load_time = 0;
  double histo_time = 0;
  double save_time = 0;

  // Set ourselves to the correct GPU
  cv::cuda::setDevice(gpu_device_id);

  int frame_size = width * height * 4 * sizeof(u8);

  auto setup_start = scanner::now();
  std::vector<cvc::GpuMat> planes;
  for (int i = 0; i < 4; ++i) {
    planes.push_back(cvc::GpuMat(height, width, CV_8UC1));
  }

  cvc::GpuMat hist(1, BINS, CV_32S);
  cvc::GpuMat out_gpu(1, BINS * 3, CV_32S);

  int64_t frame = 0;
  while (true) {
    BufferHandle buffer_handle;
    decoded_frames.pop(buffer_handle);
    u8* buffer = buffer_handle.buffer;
    int elements = buffer_handle.elements;
    if (buffer == nullptr) {
      break;
    }

    SaveHandle save_handle;
    free_output_buffers.pop(save_handle);
    u8* output_buffer = save_handle.buffer;
    save_handle.elements = elements;

    cvc::Stream stream = cvc::StreamAccessor::wrapStream(save_handle.stream);
    for (int i = 0; i < elements; ++i) {
      cvc::GpuMat image(height, width, CV_8UC3, buffer + i * frame_size);
      auto histo_start = scanner::now();
      cvc::split(image, planes, stream);
      for (int j = 0; j < 3; ++j) {
        cvc::histEven(planes[j], hist, BINS, 0, 256, stream);
        hist.copyTo(out_gpu(cv::Rect(j * BINS, 0, BINS, 1)), stream);
      }
      cudaMemcpyAsync(output_buffer + i * output_element_size, out_gpu.data,
                      output_element_size, cudaMemcpyDefault,
                      save_handle.stream);
      histo_time += scanner::nano_since(histo_start);
    }

    save_buffers.push(save_handle);
    free_buffers.push(buffer);
  }
  std::unique_lock<std::mutex> lock(TIMINGS_MUTEX);
  TIMINGS["setup"] += setup_time;
  TIMINGS["load"] += load_time;
  TIMINGS["eval"] += histo_time;
  TIMINGS["save"] += save_time;
}

void video_flow_cpu_worker(int gpu_device_id, Queue<u8 *> &free_buffers,
                           Queue<BufferHandle> &decoded_frames,
                           Queue<SaveHandle> &free_output_buffers,
                           Queue<SaveHandle> &save_buffers) {
  double load_time = 0;
  double eval_time = 0;
  double save_time = 0;
  // Set ourselves to the correct GPU
  cv::cuda::setDevice(gpu_device_id);

  int frame_size = width * height * 4 * sizeof(u8);

  std::vector<cv::Mat> gray;
  for (int i = 0; i < 2; ++i) {
    gray.emplace_back(height, width, CV_8UC1);
  }

  auto flow = cv::FarnebackOpticalFlow::create(3, 0.5, false, 15, 3, 5, 1.2, 0);

  int64_t frame = 0;
  while (true) {
    BufferHandle buffer_handle;
    decoded_frames.pop(buffer_handle);
    u8* buffer = buffer_handle.buffer;
    int elements = buffer_handle.elements;
    if (buffer == nullptr) {
      break;
    }

    SaveHandle save_handle;
    free_output_buffers.pop(save_handle);
    save_handle.elements = elements;

    u8* output_buffer = save_handle.buffer;

    printf("elements %d\n", elements);
    // Load the first frame
    auto eval_first = scanner::now();
    cv::Mat image(height, width, CV_8UC3, buffer);
    cv::cvtColor(image, gray[0], CV_BGR2GRAY);
    eval_time += scanner::nano_since(eval_first);
    bool done = false;
    for (int i = 1; i < elements; ++i) {
      int curr_idx = i % 2;
      int prev_idx = (i - 1) % 2;
      cv::Mat output_flow(height, width, CV_32FC2,
                          output_buffer + i * output_element_size);
      cv::Mat image(height, width, CV_8UC3, buffer + i * frame_size);

      auto eval_start = scanner::now();
      printf("curr %d, prev %d\n", curr_idx, prev_idx);
      cv::cvtColor(image, gray[curr_idx], CV_BGR2GRAY);
      flow->calc(gray[prev_idx], gray[curr_idx], output_flow);
      eval_time += scanner::nano_since(eval_start);
    }

    save_buffers.push(save_handle);
    free_buffers.push(buffer);
  }
  std::unique_lock<std::mutex> lock(TIMINGS_MUTEX);
  TIMINGS["load"] += load_time;
  TIMINGS["eval"] += eval_time;
  TIMINGS["save"] += save_time;
}

void video_flow_gpu_worker(int gpu_device_id, Queue<u8 *> &free_buffers,
                           Queue<BufferHandle> &decoded_frames,
                           Queue<SaveHandle> &free_output_buffers,
                           Queue<SaveHandle> &save_buffers) {
  double load_time = 0;
  double eval_time = 0;
  double save_time = 0;
  // Set ourselves to the correct GPU
  cv::cuda::setDevice(gpu_device_id);

  int frame_size = width * height * 4 * sizeof(u8);

  std::vector<cvc::GpuMat> gray;
  for (int i = 0; i < 2; ++i) {
    gray.emplace_back(height, width, CV_8UC1);
  }

  cv::Ptr<cvc::DenseOpticalFlow> flow = cvc::FarnebackOpticalFlow::create();

  int64_t frame = 0;
  while (true) {
    BufferHandle buffer_handle;
    decoded_frames.pop(buffer_handle);
    u8* buffer = buffer_handle.buffer;
    int elements = buffer_handle.elements;
    if (buffer == nullptr) {
      break;
    }

    SaveHandle save_handle;
    free_output_buffers.pop(save_handle);
    save_handle.elements = elements;

    u8* output_buffer = save_handle.buffer;
    cvc::Stream stream = cvc::StreamAccessor::wrapStream(save_handle.stream);

    // Load the first frame
    auto eval_first = scanner::now();
    cvc::GpuMat image(height, width, CV_8UC4, buffer);
    cvc::cvtColor(image, gray[0], CV_BGRA2GRAY, 0, stream);
    eval_time += scanner::nano_since(eval_first);
    bool done = false;
    for (int i = 1; i < elements; ++i) {
      int curr_idx = i % 2;
      int prev_idx = (i - 1) % 2;
      cvc::GpuMat output_flow_gpu(height, width, CV_32FC2,
                                  output_buffer + i * output_element_size);
      cvc::GpuMat image(height, width, CV_8UC4, buffer + i * frame_size);

      auto eval_start = scanner::now();
      cvc::cvtColor(image, gray[curr_idx], CV_BGRA2GRAY, 0, stream);
      flow->calc(gray[prev_idx], gray[curr_idx], output_flow_gpu, stream);
      eval_time += scanner::nano_since(eval_start);
    }

    save_buffers.push(save_handle);
    free_buffers.push(buffer);
  }
  std::unique_lock<std::mutex> lock(TIMINGS_MUTEX);
  TIMINGS["load"] += load_time;
  TIMINGS["eval"] += eval_time;
  TIMINGS["save"] += save_time;
}

void set_halide_buf(buffer_t &halide_buf, u8 *buf, size_t size) {
  halide_buf.dev = (uintptr_t) nullptr;

  // "You likely want to set the dev_dirty flag for correctness. (It will
  // not matter if all the code runs on the GPU.)"
  halide_buf.dev_dirty = true;

  i32 err = halide_cuda_wrap_device_ptr(nullptr, &halide_buf, (uintptr_t)buf);
  assert(err == 0);

  // "You'll need to set the host field of the buffer_t structs to
  // something other than nullptr as that is used to indicate bounds query
  // calls" - Zalman Stern
  halide_buf.host = (u8 *)0xdeadbeef;
}

void unset_halide_buf(buffer_t &halide_buf) {
  halide_cuda_detach_device_ptr(nullptr, &halide_buf);
}

void transform_halide(const NetDescriptor& descriptor_,
                      i32 net_width, i32 net_height,
                      u8* input_buffer, u8* output_buffer) {
  i32 net_input_width_ = net_width;
  i32 net_input_height_ = net_height;
  i32 frame_width = width;
  i32 frame_height = height;
  size_t net_input_size =
      net_input_width_ * net_input_height_ * 3 * sizeof(float);

  buffer_t input_buf = {0}, output_buf = {0};

  set_halide_buf(input_buf, input_buffer, frame_width * frame_height * 3);
  set_halide_buf(output_buf, output_buffer, net_input_size);

  // Halide has the input format x * stride[0] + y * stride[1] + c * stride[2]
  // input_buf.host = input_buffer;
  input_buf.stride[0] = 3;
  input_buf.stride[1] = frame_width * 3;
  input_buf.stride[2] = 1;
  input_buf.extent[0] = frame_width;
  input_buf.extent[1] = frame_height;
  input_buf.extent[2] = 3;
  input_buf.elem_size = 1;

  // Halide conveniently defaults to a planar format, which is what Caffe
  // expects
  output_buf.host = output_buffer;
  output_buf.stride[0] = 1;
  output_buf.stride[1] = net_input_width_;
  output_buf.stride[2] = net_input_width_ * net_input_height_;
  output_buf.extent[0] = net_input_width_;
  output_buf.extent[1] = net_input_height_;
  output_buf.extent[2] = 3;
  output_buf.elem_size = 4;

  auto func = caffe_input_transformer_gpu;
  int error = func(&input_buf, frame_width, frame_height,
                   net_input_width_, net_input_height_, descriptor_.normalize,
                   descriptor_.mean_colors[2], descriptor_.mean_colors[1],
                   descriptor_.mean_colors[0], &output_buf);
  LOG_IF(FATAL, error != 0) << "Halide error " << error;

  unset_halide_buf(input_buf);
  unset_halide_buf(output_buf);
}

void video_caffe_worker(int gpu_device_id, Queue<u8 *> &free_buffers,
                        Queue<BufferHandle> &decoded_frames,
                        Queue<SaveHandle> &free_output_buffers,
                        Queue<SaveHandle> &save_buffers) {
  double idle_time = 0;
  double load_time = 0;
  double transform_time = 0;
  double net_time = 0;
  double eval_time = 0;
  double save_time = 0;

  int frame_size = width * height * 4 * sizeof(u8);

  NetDescriptor descriptor;
  {
    std::ifstream net_file{NET_PATH};
    descriptor = scanner::descriptor_from_net_file(net_file);
  }

  // Set ourselves to the correct GPU
  CUcontext cuda_context;
  CUD_CHECK(cuDevicePrimaryCtxRetain(&cuda_context, gpu_device_id));
  Halide::Runtime::Internal::Cuda::context = cuda_context;
  halide_set_gpu_device(gpu_device_id);
  cv::cuda::setDevice(gpu_device_id);
  CU_CHECK(cudaSetDevice(gpu_device_id));
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  caffe::Caffe::SetDevice(gpu_device_id);
  std::unique_ptr<caffe::Net<float>> net;
  net.reset(new caffe::Net<float>(descriptor.model_path, caffe::TEST));
  net->CopyTrainedLayersFrom(descriptor.model_weights_path);

  const boost::shared_ptr<caffe::Blob<float>> input_blob{
    net->blob_by_name(descriptor.input_layer_names[0])};

  input_blob->Reshape({NET_BATCH_SIZE, input_blob->shape(1),
                       input_blob->shape(2), input_blob->shape(3)});


  // Setup caffe batch transformer
  caffe::TransformationParameter param;
  std::vector<float>& mean_colors = descriptor.mean_colors;
  param.set_force_color(true);
  if (descriptor.normalize) {
    param.set_scale(1.0 / 255.0);
  }
  for (int i = 0; i < mean_colors.size(); i++) {
    param.add_mean_value(mean_colors[i]);
  }
  caffe::DataTransformer<float> transformer(param, caffe::TEST);

  net->Forward();

  int net_input_width = input_blob->shape(2);  // width
  int net_input_height = input_blob->shape(3); // height
  size_t net_input_size =
      net_input_width * net_input_height * 3 * sizeof(float);

  int64_t frame = 0;
  while (true) {
    auto idle_start = scanner::now();
    BufferHandle buffer_handle;
    decoded_frames.pop(buffer_handle);
    u8* buffer = buffer_handle.buffer;
    int elements = buffer_handle.elements;
    if (buffer == nullptr) {
      break;
    }

    SaveHandle save_handle;
    free_output_buffers.pop(save_handle);
    save_handle.elements = elements;

    u8* output_buffer = save_handle.buffer;

    cvc::Stream stream = cvc::StreamAccessor::wrapStream(save_handle.stream);

    idle_time += scanner::nano_since(idle_start);

    // Load the first frame
    int64_t frame = 0;
    while (frame < elements) {
      int batch = std::min((int)(elements - frame), (int)NET_BATCH_SIZE);
      if (batch != NET_BATCH_SIZE) {
        input_blob->Reshape({batch, input_blob->shape(1), input_blob->shape(2),
                input_blob->shape(3)});
      }
      auto transform_start = scanner::now();
      for (int i = 0; i < batch; i++) {
        u8 *input = buffer + (frame + i) * frame_size;
        u8 *output =
            ((u8 *)input_blob->mutable_gpu_data()) + i * net_input_size;
        transform_halide(descriptor, net_input_width, net_input_height,
                         input, output);
      }
      transform_time += scanner::nano_since(transform_start);
      eval_time += scanner::nano_since(transform_start);

      auto net_start = scanner::now();
      net->Forward();
      net_time += scanner::nano_since(net_start);
      eval_time += scanner::nano_since(net_start);

      save_handle.stream = 0;
      // Save outputs
      auto save_start = scanner::now();
      const boost::shared_ptr<caffe::Blob<float>> output_blob{
        net->blob_by_name(descriptor.output_layer_names[0])};
      cudaMemcpyAsync(output_buffer + frame * output_element_size,
                      output_blob->gpu_data(),
                      output_element_size * batch, cudaMemcpyDefault,
                      save_handle.stream);
      frame += batch;
      save_time += scanner::nano_since(save_start);
    }

    save_buffers.push(save_handle);
    free_buffers.push(buffer);
  }
  Halide::Runtime::Internal::Cuda::context = 0;
  std::unique_lock<std::mutex> lock(TIMINGS_MUTEX);
  TIMINGS["idle"] += idle_time;
  TIMINGS["load"] += load_time;
  TIMINGS["transform"] += transform_time;
  TIMINGS["net"] += net_time;
  TIMINGS["eval"] += eval_time;
  TIMINGS["save"] += save_time;
}

int main(int argc, char** argv) {
  std::string video_list_path;
  i32 decoder_count;
  i32 eval_count;
  {
    po::variables_map vm;
    po::options_description desc("Allowed options");
    desc.add_options()("help", "Produce help message")(
        "video_list_path", po::value<std::string>()->required(),
        "Path to video file.")(

        "operation", po::value<std::string>()->required(),
        "histogram, flow, or caffe")(

        "decoder_count", po::value<int>()->required(),
        "Number of decoders")(

        "eval_count", po::value<int>()->required(),
        "Number of eval routines")(

        "width", po::value<int>()->required(),
        "Width of video.")(

        "height", po::value<int>()->required(),
        "Height of video.");
    try {
      po::store(po::parse_command_line(argc, argv, desc), vm);
      po::notify(vm);

      if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
      }

      video_list_path = vm["video_list_path"].as<std::string>();

      OPERATION = vm["operation"].as<std::string>();
      decoder_count = vm["decoder_count"].as<int>();
      eval_count = vm["eval_count"].as<int>();

      width = vm["width"].as<int>();
      height = vm["height"].as<int>();

    } catch (const po::required_option& e) {
      if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
      } else {
        throw e;
      }
    }
  }

  bool cpu_decoder = false;

  DecoderFn decoder_fn;
  WorkerFn worker_fn;
  if (OPERATION == "histogram_cpu") {
    cpu_decoder = true;
    worker_fn = video_histogram_cpu_worker;
    output_element_size = 3 * BINS * sizeof(i32);
  } else if (OPERATION == "histogram_gpu") {
    worker_fn = video_histogram_gpu_worker;
    output_element_size = 3 * BINS * sizeof(i32);
  } else if (OPERATION == "flow_cpu") {
    BATCH_SIZE = 8;      // Batch size for network
    cpu_decoder = true;
    worker_fn = video_flow_cpu_worker;
    output_element_size = 2 * height * width * sizeof(f32);
  } else if (OPERATION == "flow_gpu") {
    BATCH_SIZE = 8;      // Batch size for network
    worker_fn = video_flow_gpu_worker;
    output_element_size = 2 * height * width * sizeof(f32);
  } else if (OPERATION == "caffe") {
    worker_fn = video_caffe_worker;
    output_element_size = 1000 * sizeof(f32);
  } else {
    exit(1);
  }
  output_buffer_size = BATCH_SIZE * output_element_size;

  IS_CPU = cpu_decoder;
  if (IS_CPU) {
    decoder_fn = cpu_decoder_worker;
  } else {
    decoder_fn = decoder_worker;
  }

  // Setup decoder
  CUD_CHECK(cuInit(0));

  volatile bool done = false;

  cudaSetDevice(0);
  // Create decoded frames buffers and output buffers
  Queue<u8*> free_buffers;
  Queue<BufferHandle> decoded_frames;
  Queue<SaveHandle> free_output_buffers;
  Queue<SaveHandle> save_buffers;
  for (int i = 0; i < NUM_BUFFERS * std::max(decoder_count, eval_count);
       ++i) {
    if (IS_CPU) {
      u8 *buffer;
      buffer = (u8*)malloc(BATCH_SIZE * width * height * 4 * sizeof(u8));
      if (buffer == nullptr) {
        exit(1);
      }
      free_buffers.push(buffer);

      buffer = (u8*)malloc(output_buffer_size);
      if (buffer == nullptr) {
        exit(1);
      }
      SaveHandle handle;
      handle.buffer = buffer;
      free_output_buffers.push(handle);
    } else {
      u8 *buffer;
      CU_CHECK(cudaMalloc((void **)&buffer,
                          BATCH_SIZE * width * height * 4 * sizeof(u8)));
      free_buffers.push(buffer);

      CU_CHECK(cudaMalloc((void **)&buffer, output_buffer_size));
      SaveHandle handle;
      handle.buffer = buffer;
      CU_CHECK(
          cudaStreamCreateWithFlags(&handle.stream, cudaStreamNonBlocking));
      free_output_buffers.push(handle);
    }
  }

  // Start up workers to process videos
  std::vector<std::thread> evaluator_workers;
  for (i32 i = 0; i < eval_count; ++i) {
    evaluator_workers.emplace_back(
        worker_fn, 0, std::ref(free_buffers), std::ref(decoded_frames),
        std::ref(free_output_buffers), std::ref(save_buffers));
  }
  std::thread save_thread(save_worker, 0, std::ref(save_buffers),
                          std::ref(free_output_buffers));

  // Insert video paths into work queue
  Queue<std::string> video_paths;
  {
    std::ifstream infile(video_list_path);
    std::string line;
    while (std::getline(infile, line)) {
      if (line == "") {
        break;
      }
      std::cout << line << std::endl;
      video_paths.push(line);
    }
  }

  // Wait to make sure everything is setup first
  sleep(5);

  // Start work by setting up feeder
  std::vector<std::thread> decoder_threads;
  for (i32 i = 0; i < decoder_count; ++i) {
    decoder_threads.emplace_back(decoder_fn, 0, std::ref(video_paths),
                                 std::ref(free_buffers),
                                 std::ref(decoded_frames));
  }
  auto total_start = scanner::now();

  for (i32 i = 0; i < decoder_count; ++i) {
    video_paths.push("");
  }
  for (i32 i = 0; i < decoder_count; ++i) {
    decoder_threads[i].join();
  }
  // Tell evaluator decoder is done
  for (i32 i = 0; i < eval_count; ++i) {
    BufferHandle empty;
    empty.buffer = nullptr;
    decoded_frames.push(empty);
  }
  for (i32 i = 0; i < eval_count; ++i) {
    evaluator_workers[i].join();
  }

  SaveHandle em;
  em.buffer = nullptr;
  save_buffers.push(em);
  save_thread.join();

  sync();
  TIMINGS["total"] = scanner::nano_since(total_start);

  for (auto& kv : TIMINGS) {
    printf("TIMING: %s,%.2f\n", kv.first.c_str(), kv.second / 1000000000.0);
  }
}
