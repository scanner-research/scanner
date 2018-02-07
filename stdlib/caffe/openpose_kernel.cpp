#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <openpose/headers.hpp>

#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/cuda.h"
#include "scanner/util/opencv.h"
#include "stdlib/stdlib.pb.h"

namespace scanner {

const int POSE_KEYPOINTS = 18;
const int FACE_KEYPOINTS = 70;
const int HAND_KEYPOINTS = 21;
const int TOTAL_KEYPOINTS =
    POSE_KEYPOINTS + FACE_KEYPOINTS + 2 * HAND_KEYPOINTS;

class OpenPoseKernel : public scanner::BatchedKernel,
                       public scanner::VideoKernel {
 public:
  OpenPoseKernel(const scanner::KernelConfig& config)
    : scanner::BatchedKernel(config),
      opWrapper_{op::ThreadManagerMode::Asynchronous},
      device_(config.devices[0]) {

    proto::OpenPoseArgs args;
    args.ParseFromArray(config.args.data(), config.args.size());

    const op::WrapperStructPose wrapperStructPose{true,
                                                  {-1, 368},
                                                  {-1, -1},
                                                  op::ScaleMode::ZeroToOne,
                                                  1,
                                                  device_.id,
                                                  args.pose_num_scales(),
                                                  args.pose_scale_gap(),
                                                  op::RenderMode::None,
                                                  op::PoseModel::COCO_18,
                                                  false,
                                                  0.6,
                                                  0.7,
                                                  0,
                                                  args.model_directory(),
                                                  {op::HeatMapType::Parts},
                                                  op::ScaleMode::ZeroToOne,
                                                  0.05,
                                                  false};

    const op::WrapperStructFace wrapperStructFace{
      args.compute_face(), {368, 368}, op::RenderMode::None, 0.6, 0.7, 0.2};

    const op::WrapperStructHand wrapperStructHand{args.compute_hands(),
                                                  {368, 368},
                                                  args.hand_num_scales(),
                                                  args.hand_scale_gap(),
                                                  false,
                                                  op::RenderMode::None,
                                                  0.6,
                                                  0.7,
                                                  0.2};

    opWrapper_.configure(wrapperStructPose, wrapperStructFace,
                         wrapperStructHand, op::WrapperStructInput{},
                         op::WrapperStructOutput{});
    opWrapper_.start();
  }

  void execute(const scanner::BatchedElements& input_columns,
               scanner::BatchedElements& output_columns) override {
    auto& frame_col = input_columns[0];

    auto datumsPtr = std::make_shared<std::vector<op::Datum>>();
    for (int i = 0; i < num_rows(frame_col); ++i) {
      datumsPtr->emplace_back();
      auto& datum = datumsPtr->at(datumsPtr->size() - 1);
      CUDA_PROTECT({
        cv::cuda::GpuMat gpu_input =
            scanner::frame_to_gpu_mat(frame_col[i].as_const_frame());
        datum.cvInputData = cv::Mat(gpu_input);
      });
    }

    bool emplaced = opWrapper_.waitAndEmplace(datumsPtr);
    LOG_IF(FATAL, !emplaced) << "Failed to emplace pose work";
    std::shared_ptr<std::vector<op::Datum>> datumProcessed;
    bool popped = opWrapper_.waitAndPop(datumProcessed);
    LOG_IF(FATAL, !popped) << "Failed to pop pose results";

    for (auto& datum : *datumProcessed) {
      int num_people = datum.poseKeypoints.getSize(0);
      size_t size =
          num_people > 0 ? TOTAL_KEYPOINTS * num_people * 3 * sizeof(float) : 1;
      float* kp = new float[size / sizeof(float)];
      std::memset(kp, 0, size);
      float* curr_kp = kp;
      for (int i = 0; i < num_people; ++i) {
        std::memcpy(curr_kp,
                    datum.poseKeypoints.getPtr() + i * POSE_KEYPOINTS * 3,
                    POSE_KEYPOINTS * 3 * sizeof(float));
        curr_kp += POSE_KEYPOINTS * 3;
        if (datum.faceKeypoints.getPtr() != nullptr) {
          std::memcpy(curr_kp,
                      datum.faceKeypoints.getPtr() + i * FACE_KEYPOINTS * 3,
                      FACE_KEYPOINTS * 3 * sizeof(float));
        }
        curr_kp += FACE_KEYPOINTS * 3;
        if (datum.handKeypoints[0].getPtr() != nullptr) {
          std::memcpy(curr_kp,
                      datum.handKeypoints[0].getPtr() + i * HAND_KEYPOINTS * 3,
                      HAND_KEYPOINTS * 3 * sizeof(float));
        }
        curr_kp += HAND_KEYPOINTS * 3;
        if (datum.handKeypoints[1].getPtr() != nullptr) {
          std::memcpy(curr_kp,
                      datum.handKeypoints[1].getPtr() + i * HAND_KEYPOINTS * 3,
                      HAND_KEYPOINTS * 3 * sizeof(float));
        }
        curr_kp += HAND_KEYPOINTS * 3;
      }

      float* gpu_kp = (float*)scanner::new_buffer(device_, size);
      scanner::memcpy_buffer((scanner::u8*)gpu_kp, device_, (scanner::u8*)kp,
                             scanner::CPU_DEVICE, size);
      scanner::insert_element(output_columns[0], (scanner::u8*)gpu_kp, size);
      delete kp;
    }
  }

 private:
  scanner::DeviceHandle device_;
  op::Wrapper<std::vector<op::Datum>> opWrapper_;
};

REGISTER_OP(OpenPose).frame_input("frame").output("pose");

REGISTER_KERNEL(OpenPose, OpenPoseKernel)
    .device(scanner::DeviceType::GPU)
    .num_devices(1)
    .batch();

}  // namespace scanner
