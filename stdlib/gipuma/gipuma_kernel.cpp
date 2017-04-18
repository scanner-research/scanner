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
      : VideoKernel(config), device_(config.devices[0]), was_reset_(true) {
    set_device();

    state_.reset(new GlobalState);
    algo_params_ = new AlgorithmParameters;

    valid_.set_success(true);
    if (!args_.ParseFromArray(config.args.data(), config.args.size())) {
      RESULT_ERROR(&valid_, "GipumaKernel could not parse protobuf args");
      return;
    }

    num_cameras_ = config.input_columns.size() / 3;
    algo_params_->num_img_processed = num_cameras_;
    algo_params_->min_angle = 1.00;
    algo_params_->max_angle = 70.00;

    algo_params_->min_disparity = args_.min_disparity();
    algo_params_->max_disparity = args_.max_disparity();
    algo_params_->depthMin = args_.min_depth();
    algo_params_->depthMax = args_.max_depth();
    algo_params_->iterations = args_.iterations();
    algo_params_->box_hsize = args_.kernel_width();
    algo_params_->box_vsize = args_.kernel_height();

    algo_params_->n_best = 3;
    algo_params_->normTol = 0.1f;
  }

  ~GipumaKernel() {
    delete algo_params_;
  }

  void validate(proto::Result* result) {
    result->set_msg(valid_.msg());
    result->set_success(valid_.success());
  }

  void reset() {
    camera_params_ = CameraParameters();
    delete state_->cameras;
    state_->cameras = new CameraParameters_cu;
    was_reset_ = true;
  }

  void setup_gipuma(const BatchedColumns& input_columns) {
    i32 frame_width = frame_info_.width();
    i32 frame_height = frame_info_.height();

    // Read camera calibration matrix from columns
    for (i32 i = 0; i < num_cameras_; ++i) {
      i32 col_idx = 2 + 3 * i;
      auto& calibration_col = input_columns[col_idx];
      // Read camera parameters from camera calibration column

      u8* buffer = new_buffer(CPU_DEVICE, calibration_col.rows[0].size);
      memcpy_buffer((u8*)buffer, CPU_DEVICE, calibration_col.rows[0].buffer,
                    device_, calibration_col.rows[0].size);
      proto::Camera cam;
      cam.ParseFromArray(buffer,
                         calibration_col.rows[0].size);
      delete_buffer(CPU_DEVICE, buffer);

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

    selectViews(camera_params_, frame_width, frame_height, *algo_params_);
    i32 selected_views = camera_params_.viewSelectionSubset.size();
    assert(selected_views > 0);
    printf("Num cameras selected %d\n", selected_views);

    for (i32 i = 0; i < num_cameras_; ++i) {
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

    auto& frame_info = input_columns[1];
    check_frame_info(device_, frame_info);

    if (was_reset_) {
      setup_gipuma(input_columns);
    }

    i32 width = frame_info_.width();
    i32 height = frame_info_.height();
    i32 points_output_size = width * height * sizeof(float4);
    i32 cost_output_size = width * height * sizeof(float);

    i32 input_count = (i32)input_columns[0].rows.size();
    std::vector<cvc::GpuMat> grayscale_images_gpu(num_cameras_);
    std::vector<cv::Mat> grayscale_images(num_cameras_);
    u8* points_output_buffer =
      new_block_buffer(device_, points_output_size * input_count, input_count);
    u8* cost_output_buffer =
      new_block_buffer(device_, cost_output_size * input_count, input_count);
    for (i32 i = 0; i < input_count; ++i) {
      for (i32 c = 0; c < num_cameras_; ++c) {
        auto& frame_column = input_columns[c * 3];
        cvc::GpuMat frame_input(frame_info_.height(), frame_info_.width(),
                                CV_8UC3, frame_column.rows[i].buffer);
        assert(frame_column.rows[i].size == width * height * 3);

        grayscale_images[c] =
            cv::Mat(frame_info_.height(), frame_info_.width(), CV_8UC3);
        frame_input.download(grayscale_images[c]);
        cv::cvtColor(grayscale_images[c], grayscale_images[c], CV_BGR2GRAY, 0);
        grayscale_images[c].convertTo(grayscale_images[c], CV_32FC1);
      }

      addImageToTextureFloatGray(grayscale_images, state_->imgs,
                                 state_->cuArray);

      runcuda(*state_.get());

      // Copy estiamted points to output buffer
      cudaMemcpy(points_output_buffer + points_output_size * i,
                 state_->lines->norm4, points_output_size, cudaMemcpyDefault);
      insert_element(output_columns[0],
                 points_output_buffer + points_output_size * i,
                 points_output_size);

      // Copy costs to output buffer
      cudaMemcpy(cost_output_buffer + cost_output_size * i, state_->lines->c,
                 cost_output_size, cudaMemcpyDefault);
      insert_element(output_columns[1], cost_output_buffer + cost_output_size * i,
                 cost_output_size);

      delTexture(algo_params_->num_img_processed, state_->imgs,
                 state_->cuArray);
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
  i32 num_cameras_;
  bool was_reset_;
};

REGISTER_OP(Gipuma).variadic_inputs().outputs({"points", "cost"});

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
