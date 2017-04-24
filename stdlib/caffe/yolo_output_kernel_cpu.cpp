#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/types.pb.h"
#include "scanner/util/bbox.h"
#include "scanner/util/opencv.h"
#include "scanner/util/serialize.h"
#include "stdlib/stdlib.pb.h"

namespace scanner {

class YoloOutputKernel : public BatchedKernel, public VideoKernel {
 public:
  YoloOutputKernel(const KernelConfig& config) : BatchedKernel(config) {
    categories_ = {
        "aeroplane",   "bicycle", "bird",  "boat",      "bottle",
        "bus",         "car",     "cat",   "chair",     "cow",
        "diningtable", "dog",     "horse", "motorbike", "person",
        "pottedplant", "sheep",   "sofa",  "train",     "tvmonitor",
    };
    num_categories_ = static_cast<i32>(categories_.size());

    input_width_ = 448;
    input_height_ = 448;
    grid_width_ = 7;
    grid_height_ = 7;
    cell_width_ = input_width_ / grid_width_;
    cell_height_ = input_height_ / grid_height_;
    num_bboxes_ = 2;

    feature_vector_lengths_ = {
        grid_width_ * grid_height_ * num_categories_,  // category confidences
        grid_width_ * grid_height_ * num_bboxes_,      // objectness
        grid_width_ * grid_height_ * num_bboxes_ * 4   // bbox attributes
    };
    feature_vector_sizes_ = {
        sizeof(f32) * feature_vector_lengths_[0],
        sizeof(f32) * feature_vector_lengths_[1],
        sizeof(f32) * feature_vector_lengths_[2],
    };

    threshold_ = 0.5;
  }

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    i32 input_count = (i32)num_rows(input_columns[0]);
    for (i32 i = 0; i < input_count; ++i) {
      assert(input_columns[0][i].as_const_frame()->size() ==
             (feature_vector_sizes_[0] + feature_vector_sizes_[1] +
              feature_vector_sizes_[2]));
      f32* category_confidences_vector =
          reinterpret_cast<f32*>(input_columns[0][i].as_const_frame()->data);
      f32* objectness_vector =
          category_confidences_vector + feature_vector_lengths_[0];
      f32* bbox_vector = objectness_vector += feature_vector_lengths_[1];

      std::vector<f32> pixel_confidences(
          input_height_ * input_width_ * num_categories_, 0.0f);

      // Get bounding box data from output feature vector and turn it
      // into canonical center x, center y, width, height
      std::vector<BoundingBox> bboxes;
      for (i32 yi = 0; yi < grid_height_; ++yi) {
        for (i32 xi = 0; xi < grid_width_; ++xi) {
          for (i32 bi = 0; bi < num_bboxes_; ++bi) {
            i32 vec_offset = yi * grid_width_ + xi;

            f32 x = ((xi + bbox_vector[(vec_offset)*num_bboxes_ + bi * 4 + 0]) /
                     grid_width_) *
                    input_width_;
            f32 y = ((yi + bbox_vector[(vec_offset)*num_bboxes_ + bi * 4 + 1]) /
                     grid_height_) *
                    input_height_;

            f32 width =
                std::pow(bbox_vector[(vec_offset)*num_bboxes_ + bi * 4 + 3],
                         2) *
                input_width_;
            f32 height =
                std::pow(bbox_vector[(vec_offset)*num_bboxes_ + bi * 4 + 4],
                         2) *
                input_height_;

            std::vector<f32> category_probabilities(num_categories_);
            for (i32 c = 0; c < num_categories_; ++c) {
              f64 prob = objectness_vector[vec_offset * num_bboxes_ + bi] *
                         category_confidences_vector[vec_offset + c];
              category_probabilities[c] = prob;

              if (prob < threshold_) continue;

              for (i32 bbox_y = std::max(y - height / 2, 0.0f);
                   bbox_y < std::min(y + height / 2, (f32)input_height_);
                   ++bbox_y) {
                for (i32 bbox_x = std::max(x - width / 2, 0.0f);
                     bbox_x < std::min(x + width / 2, (f32)input_width_);
                     ++bbox_x) {
                  f32& max_confidence =
                      pixel_confidences[bbox_y * input_width_ +
                                        bbox_x * num_categories_ + c];
                  if (prob > max_confidence) {
                    max_confidence = prob;
                  }
                }
              }

              if (width < 0 || height < 0) continue;

              BoundingBox bbox;
              bbox.set_x1(x);
              bbox.set_y1(y);
              bbox.set_x2(x + width);
              bbox.set_y2(y + height);
              bbox.set_score(prob);
              bbox.set_label(c);
              bboxes.push_back(bbox);
            }
          }
        }
      }

      i32 non_thresholded_pixels = 1;
      f64 certainty = 0.0f;
      for (i32 yi = 0; yi < input_height_; ++yi) {
        for (i32 xi = 0; xi < input_width_; ++xi) {
          // For each pixel, compute difference between two most
          // confident classes
          f32 max1 = std::numeric_limits<f32>::lowest();
          f32 max2 = std::numeric_limits<f32>::lowest();
          for (i32 c = 0; c < num_categories_; ++c) {
            const f32& confidence =
                pixel_confidences[yi * input_width_ + xi * num_categories_ + c];
            if (confidence > max1) {
              max2 = max1;
              max1 = confidence;
            } else if (confidence > max2) {
              max2 = confidence;
            }
          }
          certainty += (max1 - max2);
          if (max1 > threshold_ || max2 > threshold_) {
            non_thresholded_pixels++;
          }
        }
      }

      size_t size;
      u8* buffer;
      serialize_bbox_vector(bboxes, buffer, size);
      insert_element(output_columns[0], buffer, size);
    }
  }

 private:
  std::vector<std::string> categories_;
  i32 num_categories_;
  i32 input_width_;
  i32 input_height_;
  i32 grid_width_;
  i32 grid_height_;
  i32 cell_width_;
  i32 cell_height_;
  i32 num_bboxes_;
  std::vector<i32> feature_vector_lengths_;
  std::vector<size_t> feature_vector_sizes_;

  double threshold_;
};

REGISTER_OP(YoloOutput).frame_input("caffe_output").output("bboxes");

REGISTER_KERNEL(YoloOutput, YoloOutputKernel)
    .device(DeviceType::CPU)
    .num_devices(1);
}
