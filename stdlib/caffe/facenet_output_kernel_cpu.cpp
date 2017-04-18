#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/types.pb.h"
#include "scanner/util/bbox.h"
#include "scanner/util/opencv.h"
#include "scanner/util/serialize.h"
#include "stdlib/stdlib.pb.h"

namespace scanner {

class FacenetOutputKernel : public VideoKernel {
 public:
  FacenetOutputKernel(const Kernel::Config& config) : VideoKernel(config) {
    proto::FacenetArgs args;
    args.ParseFromArray(config.args.data(), config.args.size());

    scale_ = args.scale();
    threshold_ = args.threshold();

    std::ifstream template_file{"nets/caffe_facenet/facenet_templates.bin",
                                std::ifstream::binary};
    LOG_IF(FATAL, !template_file.good()) << "Could not find template file.";
    templates_.resize(num_templates_, std::vector<float>(4));
    for (i32 t = 0; t < 25; ++t) {
      for (i32 i = 0; i < 4; ++i) {
        LOG_IF(FATAL, !template_file.good()) << "Template file not correct.";
        f32 d;
        template_file.read(reinterpret_cast<char*>(&d), sizeof(f32));
        templates_[t][i] = d;
      }
    }
  }

  void new_frame_info() override {
    net_input_width_ = std::floor(frame_info_.shape[1] * scale_);
    net_input_height_ = std::floor(frame_info_.shape[2] * scale_);

    if (net_input_width_ % 8 != 0) {
      net_input_width_ += 8 - (net_input_width_ % 8);
    };
    if (net_input_height_ % 8 != 0) {
      net_input_height_ += 8 - (net_input_height_ % 8);
    }

    grid_width_ = std::ceil(float(net_input_width_) / cell_width_);
    grid_height_ = std::ceil(float(net_input_height_) / cell_height_);

    feature_vector_lengths_ = {
      grid_width_ * grid_height_ * num_templates_,  // template probabilities
      grid_width_ * grid_height_ * num_templates_ * 4,  // template adjustments
    };
    feature_vector_sizes_ = {
      sizeof(f32) * feature_vector_lengths_[0],
      sizeof(f32) * feature_vector_lengths_[1],
    };
  }

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    auto& frame_col = input_columns[0];
    auto& orig_frame_info_col = input_columns[1];
    check_frame_info(CPU_DEVICE, orig_frame_info_col[0]);

    i32 input_count = (i32)frame_col.size();

    std::vector<i32> valid_templates = regular_valid_templates_;
    if (scale_ > 1.0) {
      valid_templates = big_valid_templates_;
    }
    // Get bounding box data from output feature vector and turn it
    // into canonical center x, center y, width, height
    for (i32 b = 0; b < input_count; ++b) {
      const Frame* frame = frame_col[b].as_const_frame();

      assert(frame->type == FrameType::F32);
      assert(frame->size() ==
             (feature_vector_sizes_[0] + feature_vector_sizes_[1]));

      std::vector<BoundingBox> bboxes;
      // Track confidence per pixel for each category so we can calculate
      // uncertainty across the frame
      f32* template_confidences = reinterpret_cast<f32*>(frame->data);
      f32* template_adjustments =
        template_confidences + feature_vector_lengths_[0];

      for (i32 t : valid_templates) {
        for (i32 xi = 0; xi < grid_width_; ++xi) {
          for (i32 yi = 0; yi < grid_height_; ++yi) {
            i32 vec_offset = xi * grid_height_ + yi;

            f32 confidence =
              template_confidences[t * grid_width_ * grid_height_ + vec_offset];
            // Apply sigmoid to confidence
            confidence = 1.0 / (1.0 + std::exp(-confidence));

            if (confidence < threshold_) continue;

            f32 x = xi * cell_width_ - 2;
            f32 y = yi * cell_height_ - 2;

            f32 width = templates_[t][2] - templates_[t][0] + 1;
            f32 height = templates_[t][3] - templates_[t][1] + 1;

            f32 dcx = template_adjustments[(num_templates_ * 0 + t) *
                                             grid_width_ * grid_height_ +
                                           vec_offset];
            x += width * dcx;

            f32 dcy = template_adjustments[(num_templates_ * 1 + t) *
                                             grid_width_ * grid_height_ +
                                           vec_offset];
            y += height * dcy;

            f32 dcw = template_adjustments[(num_templates_ * 2 + t) *
                                             grid_width_ * grid_height_ +
                                           vec_offset];
            width *= std::exp(dcw);

            f32 dch = template_adjustments[(num_templates_ * 3 + t) *
                                             grid_width_ * grid_height_ +
                                           vec_offset];
            height *= std::exp(dch);

            x = (x / net_input_width_) * frame_info_.shape[1];
            y = (y / net_input_height_) * frame_info_.shape[2];

            width = (width / net_input_width_) * frame_info_.shape[1];
            height = (height / net_input_height_) * frame_info_.shape[2];

            if (width < 0 || height < 0 || std::isnan(width) ||
                std::isnan(height) || std::isnan(x) || std::isnan(y))
              continue;

            // Clamp values to border

            BoundingBox bbox;
            bbox.set_x1(x - width / 2);
            bbox.set_y1(y - height / 2);
            bbox.set_x2(x + width / 2);
            bbox.set_y2(y + height / 2);
            bbox.set_score(confidence);

            // if (bbox.x1() < 0 || bbox.y1() < 0 || bbox.x2() >
            // frame_info_.width()
            // ||
            //     bbox.y2() > frame_info_.height())
            //   continue;

            bboxes.push_back(bbox);
          }
        }
      }

      std::vector<BoundingBox> best_bboxes;
      best_bboxes = best_nms(bboxes, 0.1);

      // Assume size of a bounding box is the same size as all bounding boxes
      size_t size;
      u8* buffer;
      serialize_bbox_vector(best_bboxes, buffer, size);
      output_columns[0].push_back(Element{buffer, size});
    }
  }

 private:
  f32 scale_;
  const std::vector<i32> regular_valid_templates_ = {
    4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24};
  const std::vector<i32> big_valid_templates_ = {4, 5, 6, 7, 8, 9, 10, 11};
  const i32 num_templates_ = 25;
  const i32 min_template_idx_ = 4;
  const i32 cell_width_ = 8;
  const i32 cell_height_ = 8;
  i32 net_input_width_;
  i32 net_input_height_;
  i32 grid_width_;
  i32 grid_height_;
  std::vector<std::vector<f32>> templates_;
  std::vector<i32> feature_vector_lengths_;
  std::vector<size_t> feature_vector_sizes_;

  double threshold_;
};

REGISTER_OP(FacenetOutput)
  .frame_input("facenet_output")
  .input("original_frame_info")
  .output("bboxes");

REGISTER_KERNEL(FacenetOutput, FacenetOutputKernel)
  .device(DeviceType::CPU)
  .num_devices(1);
}
