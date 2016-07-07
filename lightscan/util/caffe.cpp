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

#include "lightscan/util/caffe.h"
#include <opencv2/opencv.hpp>

using caffe::Blob;
using caffe::BlobProto;
using caffe::Caffe;
using caffe::Net;
using boost::shared_ptr;
using std::string;

namespace lightscan {

NetInfo load_neural_net(NetType type, int gpu_id) {
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  caffe::Caffe::SetDevice(gpu_id);

  std::string model_path;
  std::string model_weights_path;

  float* mean_image;
  int mean_width;
  int mean_height;
  int dim = 0;

  switch (type) {
  case NetType::ALEX_NET: {
    dim = 227;
    model_path =
      "features/bvlc_reference_caffenet/deploy.prototxt";
    model_weights_path =
      "features/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";

    std::string mean_proto_path =
      "features/bvlc_reference_caffenet/imagenet_mean.binaryproto";

    // Load mean image
    Blob<float> data_mean;
    BlobProto blob_proto;
    bool result = ReadProtoFromBinaryFile(mean_proto_path, &blob_proto);
    data_mean.FromProto(blob_proto);

    mean_width = 256;
    mean_height = 256;
    mean_image = new float[mean_width * mean_height * 3];
    memcpy(mean_image, data_mean.cpu_data(), sizeof(float) * 256 * 256 * 3);
    break;
  }
  case NetType::VGG: {
    dim = 224;
    model_path =
      "features/vgg19/VGG_ILSVRC_19_layers_deploy.prototxt";
    model_weights_path =
      "features/vgg19/VGG_ILSVRC_19_layers.caffemodel";

    mean_width = dim;
    mean_height = dim;
    mean_image = new float[dim * dim * 3];

    for (int i = 0; i < dim * dim; ++i) {
      size_t offset = i * 3;
      mean_image[offset + 0] = 103.939;
      mean_image[offset + 1] = 116.779;
      mean_image[offset + 2] = 123.68;
    }
    break;
  }
  case NetType::VGG_FACE: {
    dim = 224;
    model_path =
      "features/vgg_face_caffe/VGG_FACE_deploy.prototxt";
    model_weights_path =
      "features/vgg_face_caffe/VGG_FACE.caffemodel";

    mean_width = dim;
    mean_height = dim;
    mean_image = new float[dim * dim * 3];

    for (int i = 0; i < dim * dim; ++i) {
      size_t offset = i * 3;
      mean_image[offset + 0] = 93.5940;
      mean_image[offset + 1] = 104.7624;
      mean_image[offset + 2] = 129.1863;
    }
    break;
  }
  }

  // Transform mean into same size as input to network
  size_t size = mean_height * mean_width * 3 * sizeof(float);
  void* buf = malloc(size);
  memcpy(buf, mean_image, size);
  cv::Mat mean_mat = cv::Mat(mean_height, mean_width, CV_32FC3, buf);

  // Initialize our network
  caffe::Net<float>* net = new Net<float>(model_path, caffe::TEST);
  net->CopyTrainedLayersFrom(model_weights_path);
  //const shared_ptr<Blob<float>> data_blob = net->blob_by_name("data");
  //data_blob->Reshape({BATCH_SIZE, 3, dim, dim});

  NetInfo info;
  info.net = net;
  info.input_size = dim;
  info.mean_image = mean_image;
  info.mean_width = mean_width;
  info.mean_height = mean_height;
  return info;
}

}
