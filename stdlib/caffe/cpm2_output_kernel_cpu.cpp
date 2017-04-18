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

#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/types.pb.h"
#include "scanner/util/common.h"
#include "scanner/util/serialize.h"
#include "scanner/util/util.h"
#include "stdlib/stdlib.pb.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

namespace scanner {
struct ModelDescriptor {
  virtual ~ModelDescriptor() {}
  virtual const std::string& get_part_name(int n) = 0;
  virtual int num_parts() = 0;
  virtual int num_limb_seq() = 0;
  virtual const int* get_limb_seq() = 0;
  virtual const int* get_map_idx() = 0;
  virtual const std::string name() = 0;
};
namespace {

struct ColumnCompare {
  bool operator()(const std::vector<double>& lhs,
                  const std::vector<double>& rhs) const {
    return lhs[2] > rhs[2];
  }
};

struct MPIModelDescriptor : public ModelDescriptor {
  std::map<int, std::string> part2name;
  const int limbSeq[28] = {0, 1,  1,  2,  2,  3,  3,  4,  1,  5, 5, 6, 6, 7,
                           1, 14, 14, 11, 11, 12, 12, 13, 14, 8, 8, 9, 9, 10};
  const int mapIdx[28] = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                          26, 27, 28, 29, 30, 31, 38, 39, 40, 41,
                          42, 43, 32, 33, 34, 35, 36, 37};
  virtual int num_parts() { return 15; }
  virtual int num_limb_seq() { return 14; }
  virtual const int* get_limb_seq() { return limbSeq; }
  virtual const int* get_map_idx() { return mapIdx; }
  virtual const std::string name() { return "MPI_15"; }

  MPIModelDescriptor()
      : part2name{
            {0, "Head"},   {1, "Neck"},      {2, "RShoulder"}, {3, "RElbow"},
            {4, "RWrist"}, {5, "LShoulder"}, {6, "LElbow"},    {7, "LWrist"},
            {8, "RHip"},   {9, "RKnee"},     {10, "RAnkle"},   {11, "LHip"},
            {12, "LKnee"}, {13, "LAnkle"},   {14, "Chest"},    {15, "Bkg"},
        } /* End initializers */ {
    for (int l = 0; l < num_limb_seq(); l++) {
      int la = limbSeq[2 * l + 0];
      int lb = limbSeq[2 * l + 1];
      int ma = mapIdx[2 * l + 0];
      int mb = mapIdx[2 * l + 1];
      part2name[ma] = part2name[la] + "->" + part2name[lb] + "(X)";
      part2name[mb] = part2name[la] + "->" + part2name[lb] + "(Y)";
    }
  }
  virtual const std::string& get_part_name(int n) { return part2name.at(n); }
};

struct COCOModelDescriptor : public ModelDescriptor {
  std::map<int, std::string> part2name;
  int limbSeq[38] = {1, 2,  1,  5,  2,  3,  3,  4,  5,  6,  6,  7, 1,
                     8, 8,  9,  9,  10, 1,  11, 11, 12, 12, 13, 1, 0,
                     0, 14, 14, 16, 0,  15, 15, 17, 2,  16, 5,  17};
  int mapIdx[38] = {31, 32, 39, 40, 33, 34, 35, 36, 41, 42, 43, 44, 19,
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 47, 48,
                    49, 50, 53, 54, 51, 52, 55, 56, 37, 38, 45, 46};
  virtual int num_parts() { return 18; }
  virtual int num_limb_seq() { return 38 / 2; }
  virtual const int* get_limb_seq() { return limbSeq; }
  virtual const int* get_map_idx() { return mapIdx; }
  virtual const std::string name() { return "COCO_18"; }

  COCOModelDescriptor()
      : part2name{
            {0, "Nose"},   {1, "Neck"},      {2, "RShoulder"}, {3, "RElbow"},
            {4, "RWrist"}, {5, "LShoulder"}, {6, "LElbow"},    {7, "LWrist"},
            {8, "RHip"},   {9, "RKnee"},     {10, "RAnkle"},   {11, "LHip"},
            {12, "LKnee"}, {13, "LAnkle"},   {14, "REye"},     {15, "LEye"},
            {16, "REar"},  {17, "LEar"},     {18, "Bkg"},
        } /* End initializers */ {
    for (int l = 0; l < num_limb_seq(); l++) {
      int la = limbSeq[2 * l + 0];
      int lb = limbSeq[2 * l + 1];
      int ma = mapIdx[2 * l + 0];
      int mb = mapIdx[2 * l + 1];
      part2name[ma] = part2name[la] + "->" + part2name[lb] + "(X)";
      part2name[mb] = part2name[la] + "->" + part2name[lb] + "(Y)";
    }
  }
  virtual const std::string& get_part_name(int n) { return part2name.at(n); }
};
}

class CPM2OutputKernel : public VideoKernel {
 public:
  CPM2OutputKernel(const Kernel::Config& config) : VideoKernel(config) {
    proto::CPM2Args args;
    args.ParseFromArray(config.args.data(), config.args.size());
    scale_ = args.scale();
    modeldesc.reset(new MPIModelDescriptor());

    joints_.resize(max_people_ * 3 * max_num_parts_);
  }

  void new_frame_info() override {
    resize_width_ = frame_info_.shape[1] * scale_;
    resize_height_ = frame_info_.shape[2] * scale_;

    width_padding_ = (resize_width_ % 8) ? 8 - (resize_width_ % 8) : 0;
    height_padding_ = (resize_height_ % 8) ? 8 - (resize_height_ % 8) : 0;
    padded_width_ = resize_width_ + width_padding_;
    padded_height_ = resize_height_ + height_padding_;

    net_input_width_ = padded_width_;
    net_input_height_ = padded_height_;

    feature_width_ = net_input_width_;
    feature_height_ = net_input_height_;
    feature_channels_ = 44;
  }

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    assert(input_columns.size() == 3);
    i32 heatmap_idx = 0;
    i32 joints_idx = 1;
    i32 frame_info_idx = 2;

    check_frame_info(CPU_DEVICE, input_columns[frame_info_idx][0]);

    i32 input_count = (i32)num_rows(input_columns[0]);

    for (i32 b = 0; b < input_count; ++b) {
      const Frame* heatmap_frame =
        input_columns[heatmap_idx][b].as_const_frame();
      const Frame* joints_frame = input_columns[joints_idx][b].as_const_frame();
      assert(heatmap_frame->size() ==
             feature_width_ * feature_height_ * feature_channels_ *
               sizeof(f32));

      const float* heatmap = reinterpret_cast<float*>(heatmap_frame->data);
      const float* peaks = reinterpret_cast<float*>(joints_frame->data);

      std::vector<std::vector<double>> subset;
      std::vector<std::vector<std::vector<double>>> connection;
      int count =
          connect_limbs(subset, connection, heatmap, peaks, joints_.data());

      std::vector<std::vector<scanner::Point>> bodies(count);
      for (int p = 0; p < count; ++p) {
        std::vector<scanner::Point>& body_joints = bodies[p];
        for (i32 j = 0; j < num_joints_; ++j) {
          int offset = p * num_joints_ * 3 + j * 3;
          float score = joints_[offset + 2];
          float y = joints_[offset + 1];
          float x = joints_[offset + 0];

          scanner::Point joint;
          joint.set_x(x);
          joint.set_y(y);
          joint.set_score(score);
          body_joints.push_back(joint);
        }
      }
      size_t size;
      u8* buffer;
      serialize_proto_vector_of_vectors(bodies, buffer, size);
      insert_element(output_columns.at(heatmap_idx), buffer, size);
    }
  }

 protected:
  int connect_limbs(std::vector<std::vector<double>>& subset,
                    std::vector<std::vector<std::vector<double>>>& connection,
                    const float* heatmap_pointer, const float* peaks,
                    float* joints) {
    /* Parts Connection ---------------------------------------*/
    // limbSeq = [15 2; 2 1; 2 3; 3 4; 4 5; 2 6; 6 7; 7 8; 15 12; 12 13; 13 14;
    // 15
    // 9; 9 10; 10 11];
    // int limbSeq[28] = {14,1, 1,0, 1,2, 2,3, 3,4, 1,5, 5,6, 6,7, 14,11, 11,12,
    // 12,13, 14,8, 8,9, 9,10};
    // int mapIdx[14] = {27, 16, 17, 18, 19, 20, 21, 22, 15, 25, 26, 14, 23,
    // 24};

    const int NUM_PARTS = modeldesc->num_parts();
    const int* limbSeq = modeldesc->get_limb_seq();
    const int* mapIdx = modeldesc->get_map_idx();
    const int num_limb_seq = modeldesc->num_limb_seq();

    int SUBSET_CNT = NUM_PARTS + 2;
    int SUBSET_SCORE = NUM_PARTS + 1;
    int SUBSET_SIZE = NUM_PARTS + 3;

    CHECK_EQ(NUM_PARTS, 15);
    CHECK_EQ(num_limb_seq, 14);

    int peaks_offset = 3 * (max_peaks_ + 1);
    subset.clear();
    connection.clear();

    for (int k = 0; k < num_limb_seq; k++) {
      // float* score_mid = heatmap_pointer + mapIdx[k] * INIT_PERSON_NET_HEIGHT
      // *
      // INIT_PERSON_NET_WIDTH;
      const float* map_x = heatmap_pointer +
                           mapIdx[2 * k] * net_input_height_ * net_input_width_;
      const float* map_y =
          heatmap_pointer +
          mapIdx[2 * k + 1] * net_input_height_ * net_input_width_;

      const float* candA = peaks + limbSeq[2 * k] * peaks_offset;
      const float* candB = peaks + limbSeq[2 * k + 1] * peaks_offset;

      std::vector<std::vector<double>> connection_k;
      int nA = candA[0];
      int nB = candB[0];

      // add parts into the subset in special case
      if (nA == 0 && nB == 0) {
        continue;
      } else if (nA == 0) {
        for (int i = 1; i <= nB; i++) {
          std::vector<double> row_vec(SUBSET_SIZE, 0);
          row_vec[limbSeq[2 * k + 1]] =
              limbSeq[2 * k + 1] * peaks_offset + i * 3 + 2;  // store the index
          row_vec[SUBSET_CNT] =
              1;  // last number in each row is the parts number of that person
          row_vec[SUBSET_SCORE] =
              candB[i * 3 +
                    2];  // second last number in each row is the total score
          subset.push_back(row_vec);
        }
        continue;
      } else if (nB == 0) {
        for (int i = 1; i <= nA; i++) {
          std::vector<double> row_vec(SUBSET_SIZE, 0);
          row_vec[limbSeq[2 * k]] =
              limbSeq[2 * k] * peaks_offset + i * 3 + 2;  // store the index
          row_vec[SUBSET_CNT] =
              1;  // last number in each row is the parts number of that person
          row_vec[SUBSET_SCORE] =
              candA[i * 3 +
                    2];  // second last number in each row is the total score
          subset.push_back(row_vec);
        }
        continue;
      }

      std::vector<std::vector<double>> temp;
      const int num_inter = 10;

      for (int i = 1; i <= nA; i++) {
        for (int j = 1; j <= nB; j++) {
          float s_x = candA[i * 3];
          float s_y = candA[i * 3 + 1];
          float d_x = candB[j * 3] - candA[i * 3];
          float d_y = candB[j * 3 + 1] - candA[i * 3 + 1];
          float norm_vec = sqrt(pow(d_x, 2) + pow(d_y, 2));
          if (norm_vec < 1e-6) {
            continue;
          }
          float vec_x = d_x / norm_vec;
          float vec_y = d_y / norm_vec;

          float sum = 0;
          int count = 0;

          for (int lm = 0; lm < num_inter; lm++) {
            int my = round(s_y + lm * d_y / num_inter);
            int mx = round(s_x + lm * d_x / num_inter);
            int idx = my * net_input_width_ + mx;
            float score = (vec_x * map_x[idx] + vec_y * map_y[idx]);
            if (score > connect_inter_threshold_) {
              sum = sum + score;
              count++;
            }
          }
          // float score = sum / count; // + std::min((130/dist-1),0.f)

          if (count > connect_inter_min_above_threshold_) {
            // parts score + cpnnection score
            std::vector<double> row_vec(4, 0);
            row_vec[3] =
                sum / count + candA[i * 3 + 2] + candB[j * 3 + 2];  // score_all
            row_vec[2] = sum / count;
            row_vec[0] = i;
            row_vec[1] = j;
            temp.push_back(row_vec);
          }
        }
      }

      //** select the top num connection, assuming that each part occur only
      // once
      // sort rows in descending order based on parts + connection score
      if (temp.size() > 0) std::sort(temp.begin(), temp.end(), ColumnCompare());

      int num = std::min(nA, nB);
      int cnt = 0;
      std::vector<int> occurA(nA, 0);
      std::vector<int> occurB(nB, 0);

      // debug
      // 	if(k==3){
      //  cout << "connection before" << endl;
      // 	for(int i = 0; i < temp.size(); i++){
      // 	   	for(int j = 0; j < temp[0].size(); j++){
      // 	        cout << temp[i][j] << " ";
      // 	    }
      // 	    cout << endl;
      // 	}
      // 	//cout << "debug" << score_mid[ 216 * INIT_PERSON_NET_WIDTH +
      // 184] <<
      // endl;
      // }

      // cout << num << endl;
      for (int row = 0; row < temp.size(); row++) {
        if (cnt == num) {
          break;
        } else {
          int i = int(temp[row][0]);
          int j = int(temp[row][1]);
          float score = temp[row][2];
          if (occurA[i - 1] == 0 && occurB[j - 1] == 0) {  // && score> (1+thre)
            std::vector<double> row_vec(3, 0);
            row_vec[0] = limbSeq[2 * k] * peaks_offset + i * 3 + 2;
            row_vec[1] = limbSeq[2 * k + 1] * peaks_offset + j * 3 + 2;
            row_vec[2] = score;
            connection_k.push_back(row_vec);
            cnt = cnt + 1;
            // cout << "cnt: " << connection_k.size() << endl;
            occurA[i - 1] = 1;
            occurB[j - 1] = 1;
          }
        }
      }
      //   if(k==0){
      //     cout << "connection" << endl;
      //     for(int i = 0; i < connection_k.size(); i++){
      // 	   	for(int j = 0; j < connection_k[0].size(); j++){
      // 	        cout << connection_k[i][j] << " ";
      // 	    }
      // 	    cout << endl;
      // 	}
      // }
      // connection.push_back(connection_k);

      //** cluster all the joints candidates into subset based on the part
      // connection
      // initialize first body part connection 15&16
      // cout << connection_k.size() << endl;
      if (k == 0) {
        std::vector<double> row_vec(NUM_PARTS + 3, 0);
        for (int i = 0; i < connection_k.size(); i++) {
          double indexA = connection_k[i][0];
          double indexB = connection_k[i][1];
          row_vec[limbSeq[0]] = indexA;
          row_vec[limbSeq[1]] = indexB;
          row_vec[SUBSET_CNT] = 2;
          // add the score of parts and the connection
          row_vec[SUBSET_SCORE] =
              peaks[int(indexA)] + peaks[int(indexB)] + connection_k[i][2];
          subset.push_back(row_vec);
        }
      } else {
        if (connection_k.size() == 0) {
          continue;
        }
        // A is already in the subset, find its connection B
        for (int i = 0; i < connection_k.size(); i++) {
          int num = 0;
          double indexA = connection_k[i][0];
          double indexB = connection_k[i][1];

          for (int j = 0; j < subset.size(); j++) {
            if (subset[j][limbSeq[2 * k]] == indexA) {
              subset[j][limbSeq[2 * k + 1]] = indexB;
              num = num + 1;
              subset[j][SUBSET_CNT] = subset[j][SUBSET_CNT] + 1;
              subset[j][SUBSET_SCORE] = subset[j][SUBSET_SCORE] +
                                        peaks[int(indexB)] + connection_k[i][2];
            }
          }
          // if can not find partA in the subset, create a new subset
          if (num == 0) {
            std::vector<double> row_vec(SUBSET_SIZE, 0);
            row_vec[limbSeq[2 * k]] = indexA;
            row_vec[limbSeq[2 * k + 1]] = indexB;
            row_vec[SUBSET_CNT] = 2;
            row_vec[SUBSET_SCORE] =
                peaks[int(indexA)] + peaks[int(indexB)] + connection_k[i][2];
            subset.push_back(row_vec);
          }
        }
      }
      // cout << nA << " ";
    }

    // debug
    // cout << " subset " << endl;
    // for(int i = 0; i < subset.size(); i++){
    //    	for(int j = 0; j < subset[0].size(); j++){
    //         cout << subset[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    //** joints by deleteing some rows of subset which has few parts occur
    int cnt = 0;
    for (int i = 0; i < subset.size(); i++) {
      if (subset[i][SUBSET_CNT] >= connect_min_subset_cnt_ &&
          (subset[i][SUBSET_SCORE] / subset[i][SUBSET_CNT]) >
              connect_min_subset_score_) {
        for (int j = 0; j < NUM_PARTS; j++) {
          int idx = int(subset[i][j]);
          if (idx) {
            joints[cnt * NUM_PARTS * 3 + j * 3 + 2] = peaks[idx];
            joints[cnt * NUM_PARTS * 3 + j * 3 + 1] = peaks[idx - 1] *
                                                      frame_info_.shape[2] /
                                                      (float)net_input_height_;
            joints[cnt * NUM_PARTS * 3 + j * 3] =
                peaks[idx - 2] * frame_info_.shape[1] / (float)net_input_width_;
          } else {
            joints[cnt * NUM_PARTS * 3 + j * 3 + 2] = 0;
            joints[cnt * NUM_PARTS * 3 + j * 3 + 1] = 0;
            joints[cnt * NUM_PARTS * 3 + j * 3] = 0;
          }
        }
        cnt++;
        if (cnt == max_people_) {
          break;
        }
      }
    }
    return cnt;
  }

 private:
  f32 threshold_ = 0.5f;

  std::unique_ptr<ModelDescriptor> modeldesc;
  // The maximum number of joint peaks from the nms output layer
  f32 scale_;
  i32 cell_size_ = 8;
  i32 resize_width_;
  i32 resize_height_;
  i32 width_padding_;
  i32 height_padding_;
  i32 padded_width_;
  i32 padded_height_;
  i32 net_input_width_;
  i32 net_input_height_;
  i32 feature_width_;
  i32 feature_height_;
  i32 feature_channels_;

  const int max_people_ = 96;
  const int max_num_parts_ = 70;
  const int max_peaks_ = 20;
  const int num_joints_ = 15;
  int connect_min_subset_cnt_ = 3;
  float connect_min_subset_score_ = 0.4;
  float connect_inter_threshold_ = 0.01;
  int connect_inter_min_above_threshold_ = 8;
  std::vector<float> joints_;
};

REGISTER_OP(CPM2Output)
    .frame_input("cpm2_resized_map")
    .frame_input("cpm2_joints")
    .input("original_frame_info")
    .output("poses");

REGISTER_KERNEL(CPM2Output, CPM2OutputKernel)
    .device(DeviceType::CPU)
    .num_devices(1);
}
