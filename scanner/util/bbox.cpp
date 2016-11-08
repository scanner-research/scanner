#include "scanner/util/bbox.h"
#include "scanner/util/common.h"

#include <queue>

namespace scanner {

std::vector<BoundingBox> best_nms(const std::vector<BoundingBox>& boxes,
                                  f32 overlap) {
  std::vector<bool> valid(boxes.size(), true);
  auto cmp = [](std::pair<f32, i32> left, std::pair<f32, i32> right) {
    return left.first < right.first;
  };
  std::priority_queue<std::pair<f32, i32>, std::vector<std::pair<f32, i32>>,
                      decltype(cmp)>
      q(cmp);
  for (i32 i = 0; i < (i32)boxes.size(); ++i) {
    q.emplace(boxes[i].score(), i);
  }
  std::vector<i32> best;
  while (!q.empty()) {
    std::pair<f32, i32> entry = q.top();
    q.pop();
    i32 c_idx = entry.second;
    if (!valid[c_idx]) continue;

    best.push_back(c_idx);

    for (i32 i = 0; i < (i32)boxes.size(); ++i) {
      if (!valid[i]) continue;

      f32 x1 = std::max(boxes[c_idx].x1(), boxes[i].x1());
      f32 y1 = std::max(boxes[c_idx].y1(), boxes[i].y1());
      f32 x2 = std::min(boxes[c_idx].x2(), boxes[i].x2());
      f32 y2 = std::min(boxes[c_idx].y2(), boxes[i].y2());

      f32 o_w = std::max(0.0f, x2 - x1 + 1);
      f32 o_h = std::max(0.0f, y2 - y1 + 1);

      f32 box_overlap = o_w * o_h / ((boxes[i].x2() - boxes[i].x1() + 1) *
                                     (boxes[i].y2() - boxes[i].y1() + 1));

      valid[i] = box_overlap < overlap;
    }
  }

  std::vector<BoundingBox> out_boxes;
  for (i32 i : best) {
    out_boxes.push_back(boxes[i]);
  }
  return out_boxes;
}

std::vector<BoundingBox> average_nms(const std::vector<BoundingBox>& boxes,
                                     f32 overlap) {
  std::vector<BoundingBox> best_boxes;
  std::vector<bool> valid(boxes.size(), true);
  auto cmp = [](std::pair<f32, i32> left, std::pair<f32, i32> right) {
    return left.first < right.first;
  };
  std::priority_queue<std::pair<f32, i32>, std::vector<std::pair<f32, i32>>,
                      decltype(cmp)>
      q(cmp);
  for (i32 i = 0; i < (i32)boxes.size(); ++i) {
    q.emplace(boxes[i].score(), i);
  }
  std::vector<i32> best;
  while (!q.empty()) {
    std::pair<f32, i32> entry = q.top();
    q.pop();
    i32 c_idx = entry.second;
    if (!valid[c_idx]) continue;

    best.push_back(c_idx);

    const BoundingBox& current_box = boxes[c_idx];
    f64 total_weight = current_box.score();
    f64 best_x1 = current_box.x1() * current_box.score();
    f64 best_y1 = current_box.y1() * current_box.score();
    f64 best_x2 = current_box.x2() * current_box.score();
    f64 best_y2 = current_box.y2() * current_box.score();
    for (i32 i = 0; i < (i32)boxes.size(); ++i) {
      if (!valid[i]) continue;

      const BoundingBox& candidate = boxes[i];

      f32 x1 = std::max(current_box.x1(), candidate.x1());
      f32 y1 = std::max(current_box.y1(), candidate.y1());
      f32 x2 = std::min(current_box.x2(), candidate.x2());
      f32 y2 = std::min(current_box.y2(), candidate.y2());

      f32 o_w = std::max(0.0f, x2 - x1 + 1);
      f32 o_h = std::max(0.0f, y2 - y1 + 1);

      f32 box_overlap = o_w * o_h / ((candidate.x2() - candidate.x1() + 1) *
                                     (candidate.y2() - candidate.y1() + 1));

      valid[i] = box_overlap < overlap;

      // Add to average for this box
      if (!valid[i]) {
        total_weight += candidate.score();
        best_x1 += candidate.x1() * candidate.score();
        best_y1 += candidate.y1() * candidate.score();
        best_x2 += candidate.x2() * candidate.score();
        best_y2 += candidate.y2() * candidate.score();
      }
    }
    best_x1 /= total_weight;
    best_y1 /= total_weight;
    best_x2 /= total_weight;
    best_y2 /= total_weight;

    BoundingBox best_box;
    best_box.set_x1(best_x1);
    best_box.set_y1(best_y1);
    best_box.set_x2(best_x2);
    best_box.set_y2(best_y2);
    best_box.set_score(current_box.score());

    best_boxes.push_back(best_box);
  }

  return best_boxes;
}
}
