#include "scanner/engine/runtime.h"
#include "scanner/evaluators/caffe/caffe_evaluator.h"
#include "scanner/evaluators/caffe/cpm2/cpm2_input_evaluator.h"
#include "scanner/evaluators/caffe/cpm2/cpm2_parser_evaluator.h"
#include "scanner/evaluators/caffe/net_descriptor.h"
#include "scanner/evaluators/util/swizzle_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

#include <map>

namespace scanner {
namespace {

std::map<std::tuple<i32, i32>, i32> get_panel_cam_to_idx_mapping() {
  std::map<std::tuple<i32, i32>, i32> mapping;
  i32 num_panels = 20;
  i32 cams_per_panel = 24;
  i32 idx = 0;
  for (i32 p = 1; p < num_panels + 1; ++p) {
    for (i32 c = 1; c < cams_per_panel + 1; ++c) {
      mapping.insert({std::make_tuple(p, c), idx++});
    }
  }
  return mapping;
}

void split(const std::string &s, char delim, std::vector<std::string> &elems) {
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
}

PipelineDescription get_pipeline_description(const DatasetInformation &info) {
  const char *CAMERAS = std::getenv("SC_CAMERAS");
  const char *START_FRAME = std::getenv("SC_START_FRAME");
  const char *END_FRAME = std::getenv("SC_END_FRAME");

  auto mapping = get_panel_cam_to_idx_mapping();
  std::vector<i32> camera_idxs;
  {
    std::string cams(CAMERAS);
    std::vector<std::string> cam_strs;
    split(cams, ',', cam_strs);
    for (const std::string &cam_str : cam_strs) {
      std::vector<std::string> panel_and_cam;
      split(cam_str, ':', panel_and_cam);
      i32 panel = atoi(panel_and_cam[0].c_str());
      i32 cam = atoi(panel_and_cam[1].c_str());
      camera_idxs.push_back(mapping.at(std::make_tuple(panel, cam)));
    }
  }

  i32 start_frame = std::atoi(START_FRAME);
  i32 end_frame = std::atoi(END_FRAME);

  PipelineDescription desc;
  for (i32 idx : camera_idxs) {
    desc.tasks.emplace_back();
    Task &task = desc.tasks.back();
    task.table_name = std::to_string(idx);
    task.samples.emplace_back();
    TableSample &sample = task.samples.back();
    sample.job_name = base_job_name();
    sample.table_name = std::to_string(idx);
    sample.columns = {base_column_name()};
    for (i64 r = start_frame; r < end_frame; ++r) {
      sample.rows.push_back(r);
    }
  }

  NetDescriptor cpm_person_descriptor;
  {
    std::string net_descriptor_file = "features/cpm2.toml";
    std::ifstream net_file{net_descriptor_file};
    cpm_person_descriptor = descriptor_from_net_file(net_file);
  }

  f32 scale = 0.25;
  // CPM2 uses batch size for multiple scales
  i32 batch_size = 1;
  DeviceType device_type;
  VideoDecoderType decoder_type;
#ifdef HAVE_CUDA
  device_type = DeviceType::GPU;
  decoder_type = VideoDecoderType::NVIDIA;
#else
  device_type = DeviceType::CPU;
  decoder_type = VideoDecoderType::SOFTWARE;
#endif

  std::vector<std::unique_ptr<EvaluatorFactory>> &factories =
      desc.evaluator_factories;

  using namespace std::placeholders;
  CustomNetConfiguration net_config = std::bind(cpm2_net_config, scale, _1, _2);

  factories.emplace_back(
      new DecoderEvaluatorFactory(DeviceType::CPU, VideoDecoderType::SOFTWARE));
  // factories.emplace_back(
  //     new DecoderEvaluatorFactory(device_type, decoder_type));
  factories.emplace_back(new CPM2InputEvaluatorFactory(
      device_type, cpm_person_descriptor, batch_size, scale));
  factories.emplace_back(new CaffeEvaluatorFactory(
      device_type, cpm_person_descriptor, batch_size, net_config));
  factories.emplace_back(
      new CPM2ParserEvaluatorFactory(DeviceType::CPU, scale));
  factories.emplace_back(
      new SwizzleEvaluatorFactory(DeviceType::CPU, {1}, {"joint_centers"}));

  return desc;
}
}

REGISTER_PIPELINE(find_body, get_pipeline_description);
}
