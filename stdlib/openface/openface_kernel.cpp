#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"
#include "scanner/util/serialize.h"

#include "OpenFace/FaceAnalyser.h"
#include "OpenFace/GazeEstimation.h"
#include "OpenFace/LandmarkCoreIncludes.h"

#include <boost/filesystem.hpp>
#include <opencv2/imgproc.hpp>

namespace scanner {

class OpenFaceKernel : public VideoKernel {
 public:
  OpenFaceKernel(const Kernel::Config& config)
    : VideoKernel(config), clnf_model(det_parameters.model_location) {
    boost::filesystem::path au_loc_path =
      boost::filesystem::path("AU_predictors/AU_all_static.txt");
    boost::filesystem::path tri_loc_path =
      boost::filesystem::path("model/tris_68_full.txt");
    face_analyser_ =
      FaceAnalysis::FaceAnalyser(vector<cv::Vec3d>(), 0.7, 112, 112,
                                 au_loc_path.string(), tri_loc_path.string());
  }

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    auto& frame_col = input_columns[0];
    auto& frame_info_col = input_columns[1];
    auto& bbox_col = input_columns[2];
    check_frame_info(CPU_DEVICE, frame_info_col);

    i32 width = frame_info_.width();
    i32 height = frame_info_.height();
    cx = width / 2.0f;
    cy = height / 2.0f;
    fx = 500 * (width / 640.0);
    fy = 500 * (height / 480.0);
    fx = (fx + fy) / 2.0;
    fy = fx;

    i32 input_count = frame_col.rows.size();
    for (i32 b = 0; b < input_count; ++b) {
      cv::Mat img(frame_info_.height(), frame_info_.width(), CV_8UC3,
                  (u8*)frame_col.rows[b].buffer);
      cv::Mat grey;
      cv::cvtColor(img, grey, CV_BGR2GRAY);
      std::vector<BoundingBox> all_bboxes =
        deserialize_proto_vector<BoundingBox>(bbox_col.rows[b].buffer,
                                              bbox_col.rows[b].size);

      for (auto& bbox : all_bboxes) {
        f64 x1 = bbox.x1(), y1 = bbox.y1(), x2 = bbox.x2(), y2 = bbox.y2();
        f64 w = x2 - x1, h = y2 - y1;
        f64 nw = w, nh = h, dw = nw - w, dh = nh - h;
        x1 = std::max(x1 - dw / 2, 0.0);
        y1 = std::max(y1 - dh / 2, 0.0);
        x2 = std::min(x2 + dw / 2, (f64)(frame_info_.width() - 1));
        y2 = std::min(y2 + dh / 2, (f64)(frame_info_.height() - 1));
        cv::Rect_<double> cv_bbox(x1, y1, x2 - x1, y2 - y1);
        cv::rectangle(img, cv_bbox, cv::Scalar(0, 255, 0));

        bool success = LandmarkDetector::DetectLandmarksInImage(
          grey, cv_bbox, clnf_model, det_parameters);
        if (success) {
          std::vector<cv::Point2d> landmarks =
            LandmarkDetector::CalculateLandmarks(clnf_model);

          cv::Point3f gazeDirection0(0, 0, -1);
          cv::Point3f gazeDirection1(0, 0, -1);
          FaceAnalysis::EstimateGaze(clnf_model, gazeDirection0, fx, fy, cx, cy,
                                     true);
          FaceAnalysis::EstimateGaze(clnf_model, gazeDirection1, fx, fy, cx, cy,
                                     false);

          auto ActionUnits =
            face_analyser_.PredictStaticAUs(grey, clnf_model, false);

          cv::Vec6d headPose =
            LandmarkDetector::GetCorrectedPoseWorld(clnf_model, fx, fy, cx, cy);

          LandmarkDetector::DrawBox(img, headPose, cv::Scalar(255.0, 0, 0), 3,
                                    fx, fy, cx, cy);
          FaceAnalysis::DrawGaze(img, clnf_model, gazeDirection0,
                                 gazeDirection1, fx, fy, cx, cy);
          LandmarkDetector::Draw(img, clnf_model);
        }
      }

      size_t size = frame_col.rows[b].size;
      u8* output_buf = new_buffer(CPU_DEVICE, size);
      memcpy_buffer(output_buf, CPU_DEVICE, frame_col.rows[b].buffer,
                    CPU_DEVICE, size);
      insert_element(output_columns[0], output_buf, size);
    }
  }

 private:
  LandmarkDetector::FaceModelParameters det_parameters;
  LandmarkDetector::CLNF clnf_model;
  std::vector<std::string> files, depth_files, output_images,
    output_landmark_locations, output_pose_locations;
  std::vector<cv::Rect_<double>> bounding_boxes;
  int device;
  float fx, fy, cx, cy;
  FaceAnalysis::FaceAnalyser face_analyser_;
};

REGISTER_OP(OpenFace)
  .inputs({"frame", "frame_info", "faces"})
  .outputs({"features"});
REGISTER_KERNEL(OpenFace, OpenFaceKernel)
  .device(DeviceType::CPU)
  .num_devices(1);
}
