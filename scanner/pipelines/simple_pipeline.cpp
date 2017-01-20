#include "scanner/api/evaluator.h"
#include "scanner/evaluators/image_processing/blur_evaluator.h"
#include "scanner/evaluators/util/swizzle_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

namespace scanner {
namespace {
PipelineDescription get_pipeline_description(const DatasetInformation& info) {
  PipelineDescription desc;
  Sampler::strided_frames(info, desc, 2);

  Evaluator* input = make_input_evaluator({"frame", "frame_info"});

  BlurArgs args;
  char* serialized_args;
  size_t serialized_args_size;
  Evaluator *blur =
      new Evaluator("Blur", {EvalInput(input, {"frame", "frame_info"})},
                    serialized_args, serialized_args_size);

  desc.evaluators = blur;

  return desc;
}
}

REGISTER_PIPELINE(simple, get_pipeline_description);
}
