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

#include "scanner/util/queue.h"
#include "scanner/util/util.h"

#include "scanner/evaluators/caffe/net_descriptor.h"
#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"

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

#include <iostream>
#include <fstream>
#include <thread>

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

using WorkerFn= std::function<void(int, Queue<u8*>&, Queue<BufferHandle>&)>;

const int NUM_BUFFERS = 4;
const int BATCH_SIZE = 96;      // Batch size for network
const int FLOW_WORK_REDUCTION = 20;
const int BINS = 16;
const std::string NET_PATH = "features/googlenet.toml";

int GPUS_PER_NODE = 1;           // GPUs to use per node
int num_frames;
int width;
int height;

std::map<std::string, double> TIMINGS;

void decoder_feeder(int gpu_device_id, std::atomic<bool> &done,
                    u8 *encoded_data, size_t encoded_data_size,
                    VideoDecoder *decoder) {
  size_t encoded_data_offset = 0;
  while (!done.load()) {
    i32 encoded_packet_size = 0;
    const u8 *encoded_packet = NULL;
    if (encoded_data_offset < encoded_data_size) {
      encoded_packet_size =
          *reinterpret_cast<const i32 *>(encoded_data + encoded_data_offset);
      encoded_data_offset += sizeof(i32);
      encoded_packet = encoded_data + encoded_data_offset;
      encoded_data_offset += encoded_packet_size;
    }

    decoder->feed(encoded_packet, encoded_packet_size, false);

    if (encoded_packet_size == 0) {
      break;
    }
  }
}

void decoder_worker(int gpu_device_id, VideoDecoder *decoder,
                    Queue<u8 *> &free_buffers,
                    Queue<BufferHandle> &decoded_frames) {
  auto video_start = now();

  int64_t frame_size = width * height * 3 * sizeof(u8);
  int64_t frame = 0;
  while (frame < num_frames) {
    u8* buffer;
    free_buffers.pop(buffer);

    int64_t buffer_frame = 0;
    while (buffer_frame < BATCH_SIZE && frame < num_frames) {
      if (decoder->decoded_frames_buffered()) {
        // New frames
        bool more_frames = true;
        while (buffer_frame < BATCH_SIZE && more_frames) {
          u8 *decoded_buffer = buffer + buffer_frame * frame_size;
          more_frames = decoder->get_frame(decoded_buffer, frame_size);
          frame++;
          buffer_frame++;
        }
      }
    }
    BufferHandle h;
    h.buffer = buffer;
    h.elements = buffer_frame;
    decoded_frames.push(h);
  }
}

void video_histogram_worker(int gpu_device_id, Queue<u8 *> &free_buffers,
                            Queue<BufferHandle> &decoded_frames) {
  double setup_time = 0;
  double load_time = 0;
  double histo_time = 0;
  double save_time = 0;

  // Set ourselves to the correct GPU
  cv::cuda::setDevice(gpu_device_id);

  int frame_size = width * height * 3 * sizeof(u8);

  auto setup_start = scanner::now();
  std::vector<cvc::GpuMat> planes;
  for (int i = 0; i < 4; ++i) {
    planes.push_back(cvc::GpuMat(height, width, CV_8UC1));
  }

  cvc::GpuMat hist = cvc::GpuMat(1, BINS, CV_32S);
  cvc::GpuMat out_gpu = cvc::GpuMat(1, BINS * 3, CV_32S);
  cv::Mat out = cv::Mat(1, BINS * 3, CV_32S);

  int64_t frame = 0;
  while (true) {
    BufferHandle buffer_handle;
    decoded_frames.pop(buffer_handle);
    u8* buffer = buffer_handle.buffer;
    int elements = buffer_handle.elements;
    if (buffer == nullptr) {
      break;
    }

    for (int i = 0; i < elements; ++i) {
      cvc::GpuMat image(height, width, CV_8UC3, buffer + i * frame_size);
      auto histo_start = scanner::now();
      cvc::split(image, planes);
      for (int j = 0; j < 3; ++j) {
        cvc::histEven(planes[j], hist, BINS, 0, 256);
        hist.copyTo(out_gpu(cv::Rect(j * BINS, 0, BINS, 1)));
      }
      out_gpu.download(out);
      histo_time += scanner::nano_since(histo_start);
    }

    free_buffers.push(buffer);
  }
  TIMINGS["setup"] = setup_time;
  TIMINGS["load"] = load_time;
  TIMINGS["eval"] = histo_time;
  TIMINGS["save"] = save_time;
}

void video_flow_worker(int gpu_device_id, Queue<u8 *> &free_buffers,
                       Queue<BufferHandle> &decoded_frames) {
  double load_time = 0;
  double eval_time = 0;
  double save_time = 0;
  // Set ourselves to the correct GPU
  cv::cuda::setDevice(gpu_device_id);

  int frame_size = width * height * 3 * sizeof(u8);

  std::vector<cvc::GpuMat> inputs;
  for (int i = 0; i < 2; ++i) {
    inputs.emplace_back(height, width, CV_8UC3);
  }
  std::vector<cvc::GpuMat> gray;
  for (int i = 0; i < 2; ++i) {
    gray.emplace_back(height, width, CV_8UC1);
  }
  cvc::GpuMat output_flow_gpu(height, width, CV_32FC2);
  cv::Mat output_flow(height, width, CV_32FC2);

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

    // Load the first frame
    auto eval_first = scanner::now();
    cvc::GpuMat image(height, width, CV_8UC3, buffer);
    cvc::cvtColor(image, gray[0], CV_BGR2GRAY);
    eval_time += scanner::nano_since(eval_first);
    bool done = false;
    for (int i = 1; i < elements; ++i) {
      int curr_idx = i % 2;
      int prev_idx = (i - 1) % 2;

      auto eval_start = scanner::now();
      cvc::GpuMat image(height, width, CV_8UC3, buffer + i * frame_size);
      cvc::cvtColor(image, gray[curr_idx], CV_BGR2GRAY);
      flow->calc(gray[prev_idx], gray[curr_idx], output_flow_gpu);
      output_flow_gpu.download(output_flow);
      eval_time += scanner::nano_since(eval_start);
    }

    free_buffers.push(buffer);
  }
  TIMINGS["load"] = load_time;
  TIMINGS["eval"] = eval_time;
  TIMINGS["save"] = save_time;
}

// void video_caffe_worker(int gpu_device_id, Queue<int64_t>& work_items) {
//   double load_time = 0;
//   double transform_time = 0;
//   double net_time = 0;
//   double eval_time = 0;
//   double save_time = 0;

//   NetDescriptor descriptor;
//   {
//     std::ifstream net_file{NET_PATH};
//     descriptor = scanner::descriptor_from_net_file(net_file);
//   }

//   // Set ourselves to the correct GPU
//   cv::cuda::setDevice(gpu_device_id);
//   CU_CHECK(cudaSetDevice(gpu_device_id));
//   caffe::Caffe::set_mode(caffe::Caffe::GPU);
//   caffe::Caffe::SetDevice(gpu_device_id);
//   std::unique_ptr<caffe::Net<float>> net;
//   net.reset(new caffe::Net<float>(descriptor.model_path, caffe::TEST));
//   net->CopyTrainedLayersFrom(descriptor.model_weights_path);

//   const boost::shared_ptr<caffe::Blob<float>> input_blob{
//     net->blob_by_name(descriptor.input_layer_names[0])};

//   // const boost::shared_ptr<caffe::Blob<float>> output_blob{
//   //   net.blob_by_name(net_descriptor.output_layer_name)};

//   int net_input_width = input_blob->shape(2); // width
//   int net_input_height = input_blob->shape(3); // height

//   // Setup caffe batch transformer
//   caffe::TransformationParameter param;
//   std::vector<float>& mean_colors = descriptor.mean_colors;
//   param.set_force_color(true);
//   if (descriptor.normalize) {
//     param.set_scale(1.0 / 255.0);
//   }
//   for (int i = 0; i < mean_colors.size(); i++) {
//     param.add_mean_value(mean_colors[i]);
//   }
//   caffe::DataTransformer<float> transformer(param, caffe::TEST);

//   cv::VideoCapture video;
//   while (true) {
//     int64_t work_item_index;
//     work_items.pop(work_item_index);

//     if (work_item_index == -1) {
//       break;
//     }

//     const std::string& path = PATHS[work_item_index];
//     video.open(path);
//     assert(video.isOpened());
//     int width = (int)video.get(CV_CAP_PROP_FRAME_WIDTH);
//     int height = (int)video.get(CV_CAP_PROP_FRAME_HEIGHT);
//     assert(width != 0 && height != 0);

//     std::ofstream outfile(output_path(work_item_index),
//                           std::fstream::binary | std::fstream::trunc);
//     assert(outfile.good());

//     // Load the first frame
//     cv::Mat input;
//     std::vector<cv::Mat> images(BATCH_SIZE);
//     int64_t frame = 0;
//     bool done = false;
//     while (!done) {
//       int b;
//       images.resize(BATCH_SIZE);
//       for (b = 0; b < BATCH_SIZE; b++) {
//         auto load_start = scanner::now();
//         bool valid_frame = video.read(input);
//         load_time += scanner::nano_since(load_start);
//         if (!valid_frame) {
//           done = true;
//           break;
//         }
//         auto transform_start = scanner::now();
//         cv::resize(input, images[b],
//                    cv::Size(net_input_width, net_input_height), 0, 0,
//                    cv::INTER_LINEAR);
//         cv::cvtColor(images[b], images[b], CV_RGB2BGR);
//         transform_time += scanner::nano_since(transform_start);
//         eval_time += scanner::nano_since(transform_start);
//       }

//       int batch = b;
//       images.resize(batch);
//       input_blob->Reshape({batch, input_blob->shape(1), input_blob->shape(2),
//                            input_blob->shape(3)});

//       auto transform_start = scanner::now();
//       transformer.Transform(images, input_blob.get());
//       transform_time += scanner::nano_since(transform_start);
//       eval_time += scanner::nano_since(transform_start);

//       auto net_start = scanner::now();
//       net->Forward();
//       net_time += scanner::nano_since(net_start);
//       eval_time += scanner::nano_since(net_start);

//       // Save outputs
//       auto save_start = scanner::now();
//       const boost::shared_ptr<caffe::Blob<float>> output_blob{
//         net->blob_by_name(descriptor.output_layer_names[0])};
//       outfile.write((char*)output_blob->cpu_data(),
//                     output_blob->count() * sizeof(float));

//       frame += batch;
//       save_time += scanner::nano_since(save_start);
//     }
//   }
//   TIMINGS["load"] = load_time;
//   TIMINGS["transform"] = transform_time;
//   TIMINGS["net"] = net_time;
//   TIMINGS["eval"] = eval_time;
//   TIMINGS["save"] = save_time;
// }

int main(int argc, char** argv) {
  std::string video_path;
  std::string operation;
  {
    po::variables_map vm;
    po::options_description desc("Allowed options");
    desc.add_options()("help", "Produce help message")(
        "video_path", po::value<std::string>()->required(),
        "Path to video file.")(

        "operation", po::value<std::string>()->required(),
        "histogram, flow, or caffe")(

        "frames", po::value<int>()->required(),
        "Number of frames to process")(

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

      video_path = vm["video_path"].as<std::string>();

      operation = vm["operation"].as<std::string>();

      num_frames = vm["frames"].as<int>();
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

  WorkerFn worker_fn;
  if (operation == "histogram") {
    worker_fn = video_histogram_worker;
  } else if (operation == "flow") {
    worker_fn = video_flow_worker;
  } else if (operation == "caffe") {
    exit(1);
    //worker_fn = video_caffe_worker;
  }

  // Read in video bytes
  std::vector<u8> video_bytes;
  {
    std::fstream fs(video_path, std::fstream::in | std::fstream::binary |
                                    std::fstream::ate);
    std::fstream::pos_type pos = fs.tellg();
    video_bytes.resize(pos);

    fs.seekg(0, std::fstream::beg);
    fs.read((char*)video_bytes.data(), pos);
  }

  // Setup decoder
  CUD_CHECK(cuInit(0));
  CUcontext cuda_context;
  CUD_CHECK(cuDevicePrimaryCtxRetain(&cuda_context, 0));
  VideoDecoder *decoder =
      new PeakVideoDecoder(0, DeviceType::GPU, cuda_context);

  scanner::InputFormat format(width, height);
  decoder->configure(format);

  std::atomic<bool> done(false);
  Queue<u8*> free_buffers;
  Queue<BufferHandle> decoded_frames;
  for (int i = 0; i < NUM_BUFFERS; ++i) {
    u8* buffer;
    CU_CHECK(cudaMalloc((void **)&buffer,
                        BATCH_SIZE * width * height * 3 * sizeof(u8)));
    free_buffers.push(buffer);
  }

  // Start up workers to process videos
  std::thread decoder_thread(decoder_worker, 0, decoder, std::ref(free_buffers),
                             std::ref(decoded_frames));
  std::thread evaluator_worker(worker_fn, 0, std::ref(free_buffers),
                               std::ref(decoded_frames));

  // Start work by setting up feeder
  auto total_start = scanner::now();
  std::thread decoder_feeder_thread(decoder_feeder, 0, std::ref(done),
                                    video_bytes.data(), video_bytes.size(),
                                    decoder);

  decoder_thread.join();
  // Tell feeder the decoder thread is done
  done = true;
  decoder_feeder_thread.join();
  // Tell evaluator decoder is done
  BufferHandle empty;
  empty.buffer = nullptr;
  decoded_frames.push(empty);
  evaluator_worker.join();

  TIMINGS["total"] = scanner::nano_since(total_start);

  for (auto& kv : TIMINGS) {
    printf("TIMING: %s,%.2f\n", kv.first.c_str(), kv.second / 1000000000.0);
  }
}
