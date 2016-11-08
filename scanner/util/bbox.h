#pragma once

#include "scanner/evaluators/types.pb.h"

namespace scanner {

std::vector<BoundingBox> best_nms(const std::vector<BoundingBox>& boxes,
                                  f32 overlap);

std::vector<BoundingBox> average_nms(const std::vector<BoundingBox>& boxes,
                                     f32 overlap);
}
