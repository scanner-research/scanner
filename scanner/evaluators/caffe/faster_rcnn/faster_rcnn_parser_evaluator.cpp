#include "scanner/evaluators/caffe/faster_rcnn/faster_rcnn_parser_evaluator.h"
#include "scanner/evaluators/serialize.h"
#include "scanner/util/bbox.h"

namespace scanner {

#define CLASSES 21
#define SCORE_THRESHOLD 0.8

void FasterRCNNParserEvaluator::evaluate(
    const std::vector<std::vector<u8*>>& input_buffers,
    const std::vector<std::vector<size_t>>& input_sizes,
    std::vector<std::vector<u8*>>& output_buffers,
    std::vector<std::vector<size_t>>& output_sizes) {
  assert(input_buffers.size() == 3);

  i32 input_count = input_buffers[0].size();
  const std::vector<u8*> cls_prob = input_buffers[0], rois = input_buffers[1],
                         fc7 = input_buffers[2];

  for (i32 i = 0; i < input_count; ++i) {
    i32 proposal_count = input_sizes[1][i] / (5 * sizeof(f32));

    std::vector<BoundingBox> bboxes;
    for (i32 j = 0; j < proposal_count; ++j) {
      f32* roi = (f32*)(rois[i] + (j * 5 * sizeof(f32)));
      f32 x1 = roi[1], y1 = roi[2], x2 = roi[3], y2 = roi[4];

      BoundingBox bbox;
      bbox.set_x1(x1);
      bbox.set_y1(y1);
      bbox.set_x2(x2);
      bbox.set_y2(y2);

      f32 max_score;
      for (i32 cls = 1; cls < CLASSES; ++cls) {
        f32* scores = (f32*)(cls_prob[i] + (j * CLASSES * sizeof(f32)));
        f32 score = scores[cls];
        if (score > SCORE_THRESHOLD) {
          bbox.set_score(score);
          bboxes.push_back(bbox);
          break;
        }
      }
    }

    std::vector<BoundingBox> best_bboxes;
    best_bboxes = best_nms(bboxes, 0.3);

    size_t size;
    u8* buffer;
    serialize_bbox_vector(best_bboxes, buffer, size);
    output_buffers[0].push_back(buffer);
    output_sizes[0].push_back(size);
  }
}

EvaluatorCapabilities FasterRCNNParserEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = DeviceType::CPU;
  caps.max_devices = 1;
  caps.warmup_size = 0;
  return caps;
}

std::vector<std::string> FasterRCNNParserEvaluatorFactory::get_output_names() {
  return {"bboxes"};
}

Evaluator* FasterRCNNParserEvaluatorFactory::new_evaluator(
    const EvaluatorConfig& config) {
  return new FasterRCNNParserEvaluator;
}
}
