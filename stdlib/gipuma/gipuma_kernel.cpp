#include "scanner/api/op.h"
#include "scanner/api/kernel.h"
#include "scanner/util/memory.h"
#include "scanner/util/cuda.h"
#include <opencv2/imgproc.hpp>
#include "scanner/util/opencv.h"
#include "stdlib/stdlib.pb.h"

#include "gipuma/cameraGeometryUtils.h"
#include "gipuma/gipuma.h"

namespace scanner {

class GipumaKernel : public VideoKernel {
public:
  GipumaKernel(const Kernel::Config &config)
      : VideoKernel(config), device_(config.devices[0]) {
    set_device();

    state_.reset(new GlobalState);
    algo_params_ = new AlgorithmParameters;

    valid_.set_success(true);
    if (!args_.ParseFromArray(config.args.data(), config.args.size())) {
      RESULT_ERROR(&valid_, "GipumaKernel could not parse protobuf args");
      return;
    }

    algo_params_->num_img_processed = args_.cameras_size();
    algo_params_->min_angle = 1.00;
    algo_params_->max_angle = 90.00;

    algo_params_->min_disparity = args_.min_disparity();
    algo_params_->max_disparity = args_.max_disparity();
    algo_params_->depthMin = args_.min_depth();
    algo_params_->depthMax = args_.max_depth();
    algo_params_->iterations = args_.iterations();
    algo_params_->box_hsize = args_.kernel_width();
    algo_params_->box_vsize = args_.kernel_height();

    // Read camera calibration matrix from args
    for (auto& cam : args_.cameras()) {
      camera_params_.cameras.emplace_back();
      auto& c = camera_params_.cameras.back();
      for (i32 i = 0; i < 3; ++i) {
        for (i32 j = 0; j < 4; ++j) {
          i32 idx = i * 4 + j;
          c.P(i, j) = cam.p(idx);
        }
      }
    }
    camera_params_ = getCameraParameters(*(state_->cameras), camera_params_);

  }

  ~GipumaKernel() {
    delete algo_params_;
  }

  void new_frame_info() {
    i32 frame_width = frame_info_.width();
    i32 frame_height = frame_info_.height();

    set_device();

    selectViews(camera_params_, frame_width, frame_height, *algo_params_);
    i32 selected_views = camera_params_.viewSelectionSubset.size();
    assert(selected_views > 0);

    for (i32 i = 0; i < 2; ++i) {
      camera_params_.cameras[i].depthMin = algo_params_->depthMin;
      camera_params_.cameras[i].depthMax = algo_params_->depthMax;
      state_->cameras->cameras[i].depthMin = algo_params_->depthMin;
      state_->cameras->cameras[i].depthMax = algo_params_->depthMax;

      algo_params_->min_disparity = disparityDepthConversion(
          camera_params_.f, camera_params_.cameras[i].baseline,
          camera_params_.cameras[i].depthMax);

      algo_params_->max_disparity = disparityDepthConversion(
          camera_params_.f, camera_params_.cameras[i].baseline,
          camera_params_.cameras[i].depthMin);
    }

    for (i32 i = 0; i < selected_views; ++i) {
      state_->cameras->viewSelectionSubset[i] =
          camera_params_.viewSelectionSubset[i];
    }

    state_->params = algo_params_;
    state_->cameras->viewSelectionSubsetNumber = selected_views;

    state_->cameras->cols = frame_width;
    state_->cameras->rows = frame_height;
    algo_params_->cols = frame_width;
    algo_params_->rows = frame_height;

    // Resize lines
    state_->lines->n = frame_height * frame_width;
    state_->lines->resize(frame_height * frame_width);
    state_->lines->s = frame_width;
    state_->lines->l = frame_width;
  }

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    set_device();

    auto& left_frame = input_columns[0];
    auto& left_frame_info = input_columns[1];
    auto& right_frame = input_columns[2];
    auto& right_frame_info = input_columns[3];
    check_frame_info(device_, left_frame_info);

    i32 width = frame_info_.width();
    i32 height = frame_info_.height();

    i32 input_count = (i32)input_columns[0].rows.size();
    std::vector<cvc::GpuMat> grayscale_images_gpu(2);
    std::vector<cvc::GpuMat> grayscale_images_gpu_f32(2);
    std::vector<cv::Mat> grayscale_images(2);
    for (i32 i = 0; i < input_count; ++i) {
      cvc::GpuMat left_input(frame_info_.height(), frame_info_.width(), CV_8UC3,
                             left_frame.rows[i].buffer);
      cvc::GpuMat right_input(frame_info_.height(), frame_info_.width(),
                              CV_8UC3, right_frame.rows[i].buffer);
      assert(left_frame.rows[i].size == width * height * 3);
      assert(right_frame.rows[i].size == width * height * 3);

      grayscale_images[0] =
          cv::Mat(frame_info_.height(), frame_info_.width(), CV_8UC3);
      grayscale_images[1] =
          cv::Mat(frame_info_.height(), frame_info_.width(), CV_8UC3);
      left_input.download(grayscale_images[0]);
      right_input.download(grayscale_images[1]);
      cv::cvtColor(grayscale_images[0], grayscale_images[0], CV_BGR2GRAY, 0);
      cv::cvtColor(grayscale_images[1], grayscale_images[1], CV_BGR2GRAY, 0);
      grayscale_images[0].convertTo(grayscale_images[0], CV_32FC1);
      grayscale_images[1].convertTo(grayscale_images[1], CV_32FC1);

      addImageToTextureFloatGray(grayscale_images, state_->imgs,
                                 state_->cuArray);
      runcuda(*state_.get());
      cv::Mat_<float> disparity = cv::Mat::zeros(height, width, CV_32FC1);
      cv::Mat_<cv::Vec3f> norm0 = cv::Mat::zeros(height, width, CV_32FC3);
      float min = 10000000000;
      float max = 0;
      for (int i = 0; i < grayscale_images[0].cols; i++) {
        for (int j = 0; j < grayscale_images[0].rows; j++) {
          int center = i + grayscale_images[0].cols * j;
          //float4 n = state_->lines->norm4[center];
          //norm0(j, i) = Vec3f(n.x, n.y, n.z);
          //disparity(j, i) = state_->lines->norm4[center].w;
          disparity(j, i) = state_->lines->norm4[center].w;
          float4 n = state_->lines->norm4[center];
          norm0(j, i) = cv::Vec3f(n.x, n.y, n.z);
        }
      }

      static int fr = 0;
      cv::Mat display;
      cv::normalize(disparity, display, 0, 65535, cv::NORM_MINMAX, CV_16U);
      char fname[256];
      sprintf(fname, "depth%05d.png", fr);
      cv::imwrite(std::string(fname), display);

      cv::Mat disp;
      norm0.convertTo(disp, CV_16U, 32767, 32767);
      cv::cvtColor(disp, disp, CV_RGB2BGR);
      cv::normalize(disp, disp, 0, 65535, cv::NORM_MINMAX, CV_16U);
      sprintf(fname, "normals%05d.png", fr++);
      cv::imwrite(std::string(fname), disp);

      std::vector<cv::Point3f> points;
      points.push_back(cv::Point3f(150, 150, 400));
      delTexture(algo_params_->num_img_processed, state_->imgs, state_->cuArray);
      u8* buf = new_buffer(device_, 1);
      INSERT_ROW(output_columns[0], buf, 1);

      printf("row\n");
    }
  }

  void set_device() {
    cudaSetDevice(device_.id);
    cvc::setDevice(device_.id);
  }

private:
  DeviceHandle device_;
  proto::Result valid_;
  proto::GipumaArgs args_;
  CameraParameters camera_params_;
  AlgorithmParameters* algo_params_;
  std::unique_ptr<GlobalState> state_;
};

REGISTER_OP(Gipuma)
    .inputs({
        "frame0", "frame_info0", "frame1", "frame_info1",
    })
    .outputs({"disparity"});

REGISTER_KERNEL(Gipuma, GipumaKernel)
    .device(DeviceType::GPU)
    .num_devices(1);
}

// {
//   cv::Mat left_cam(3, 3, CV_32F);
//   cv::Mat left_rot(3, 3, CV_64F);
//   cv::Mat left_t(3, 1, CV_32F);
//   // left_cam.at<float>(0, 0) = 745.606;
//   // left_cam.at<float>(1, 0) = 0;
//   // left_cam.at<float>(2, 0) = 0;
//   // left_cam.at<float>(0, 1) = 0;
//   // left_cam.at<float>(1, 1) = 746.049;
//   // left_cam.at<float>(2, 1) = 0;
//   // left_cam.at<float>(0, 2) = 374.278;
//   // left_cam.at<float>(1, 2) = 226.198;
//   // left_cam.at<float>(2, 2) = 1;

//   // left_rot.at<float>(0, 0) = 0.968079;
//   // left_rot.at<float>(1, 0) = -0.0488040;
//   // left_rot.at<float>(2, 0) = 0.245846;
//   // left_rot.at<float>(0, 1) = 0.0286566;
//   // left_rot.at<float>(1, 1) = 0.9959522125;
//   // left_rot.at<float>(2, 1) = 0.0852241737;
//   // left_rot.at<float>(0, 2) = -0.2490111439;
//   // left_rot.at<float>(1, 2) = -0.0754808267;
//   // left_rot.at<float>(2, 2) = 0.965554812;

//   // left_t.at<float>(0, 0) = -49.73322;
//   // left_t.at<float>(1, 0) = 142.7355424;
//   // left_t.at<float>(2, 0) = 288.2857244;
//   cv::decomposeProjectionMatrix(camera_params_.cameras[0].P, left_cam, left_rot,
//                                 left_t);
//   left_rot.convertTo(left_rot, CV_64F);
//   cv::Mat_<float> t(3, 1);
//   t(0, 0) = left_t.at<float>(0, 0);
//   t(1, 0) = left_t.at<float>(1, 0);
//   t(2, 0) = left_t.at<float>(2, 0);
//   std::vector<cv::Point2f> project_points;
//   cv::Mat dist(5, 1, CV_32F);
//   dist.at<float>(0) = -0.319142;
//   dist.at<float>(1) = 0.0562943;
//   dist.at<float>(2) = -0.000819917;
//   dist.at<float>(3) = 0.000917149;
//   dist.at<float>(4) = 0.054014;
//   cv::projectPoints(points, left_rot, t, left_cam, dist, project_points);
//   cv::circle(grayscale_images[0], project_points[0], 10, cv::Scalar(255, 0, 0),
//              3);
//   cv::imwrite("left.png", grayscale_images[0]);
// }
