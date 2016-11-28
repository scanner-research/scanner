#include "scanner/engine.h"
#include "scanner/eval/evaluator.h"
#include "scanner/eval/evaluator_factory.h"
#include "scanner/evaluators/video/decoder_evaluator.h"
#include "scanner/util/opencv.h"

#include <Eigen/Core>

namespace scanner {

template <typename Scalar, typename Vector, typename Container,
          typename distFunction>
void geometric_median(const Container& X, Vector& geo_median,
                      distFunction distance, int iterations = 10) {
  size_t dim = geo_median.size();
  size_t N = X.size();
  if (N < 3) return;

  // initial guess
  std::vector<Vector> A(2.0, (X[0] + X[1]) / Scalar(2.0));

  for (int it = 0; it < iterations; it++) {
    Vector numerator;
    for (size_t i = 0; i < dim; i++) numerator[i] = 0.;
    Scalar denominator = 0.;

    int t = it % 2;

    for (int n = 0; n < N; n++) {
      Scalar dist = distance(X[n], A[t]);

      if (dist != 0) {
        numerator += X[n] / dist;
        denominator += 1.0 / dist;
      }
    }

    A[1 - t] = numerator / denominator;
  }

  geo_median = A[iterations % 2];
}

class MedianEvaluator : public Evaluator {
 public:
  MedianEvaluator(DeviceType device_type, i32 device_id)
      : device_type_(device_type), device_id_(device_id) {}

  ~MedianEvaluator() {}

  void evaluate(const BatchedColumns& input_columns,
                BatchedColumns& output_columns) {
    i32 input_count = (i32)input_columns[0].rows.size();
    i32 out_size = 3 * sizeof(u8);

    struct distFunction {
      double operator()(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
        return (a - b).lpNorm<2>();
      }
    };

    for (i32 i = 0; i < input_count; ++i) {
      cv::Mat img = bytesToImage(input_columns[0].rows[i].buffer, metadata_);
      img.convertTo(img, CV_32FC3);

      std::vector<Eigen::Vector3d> X;
      for (i32 row = 0; row < img.rows; row += 16) {
        for (i32 col = 0; col < img.cols; col += 16) {
          cv::Vec3f pt = img.at<cv::Vec3f>(row, col);
          X.push_back(Eigen::Vector3d(pt.val[0], pt.val[1], pt.val[2]));
        }
      }

      Eigen::Vector3d median(0, 0, 0);

      geometric_median<float>(X, median, distFunction());
      // LOG(INFO) << X.size() << " " << median;

      u8* out_buffer = new u8[out_size];
      for (i32 j = 0; j < 3; ++j) {
        out_buffer[j] = (u8)median[j];
      }

      INSERT_ROW(output_columns[0], out_buffer, out_size);
    }
  }

 private:
  DeviceType device_type_;
  i32 device_id_;
};

class MedianEvaluatorFactory : public EvaluatorFactory {
 public:
  MedianEvaluatorFactory(DeviceType device_type) : device_type_(device_type) {}

  EvaluatorCapabilities get_capabilities() {
    EvaluatorCapabilities caps;
    caps.device_type = device_type_;
    caps.max_devices = 1;
    caps.warmup_size = 0;
    return caps;
  }

  std::vector<std::string> get_output_names() { return {"median"}; }

  Evaluator* new_evaluator(const EvaluatorConfig& config) {
    return new MedianEvaluator(device_type_, config.device_ids[0]);
  }

 private:
  DeviceType device_type_;
};

namespace {
PipelineDescription get_pipeline_description(
    const DatasetMetadata& dataset_meta,
    const std::vector<DatasetItemMetadata>& item_metas) {
  PipelineDescription desc;
  desc.input_columns = {"frame"};

  DeviceType device_type;
  VideoDecoderType decoder_type;

#ifdef HAVE_CUDA
  device_type = DeviceType::GPU;
  decoder_type = VideoDecoderType::NVIDIA;
#else
  device_type = DeviceType::CPU;
  decoder_type = VideoDecoderType::SOFTWARE;
#endif

  std::vector<std::unique_ptr<EvaluatorFactory>>& factories =
      desc.evaluator_factories;

  factories.emplace_back(
      new DecoderEvaluatorFactory(device_type, decoder_type));
  factories.emplace_back(new MedianEvaluatorFactory(DeviceType::CPU));

  return desc;
}
}

REGISTER_PIPELINE(median, get_pipeline_description);
}
