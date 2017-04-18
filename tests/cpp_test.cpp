#include "scanner/api/database.h"
#include "scanner/api/op.h"
#include "scanner/util/fs.h"
#include "stdlib/stdlib.pb.h"

#include <gtest/gtest.h>

namespace scanner {

// Fixtures are taken down after every test, so to avoid-redownloading and
// ingesting the files, we use static globals.
static bool downloaded = false;
static std::string db_path = "";

class ScannerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create database
    if (!downloaded) {
      scanner::temp_dir(db_path);
    }
    sc_.reset(storehouse::StorageConfig::make_posix_config());
    std::string master_port = "5001";
    std::string worker_port = "5002";
    std::string master_address = "localhost:" + master_port;
    db_ = new scanner::Database(sc_.get(), db_path, master_address);

    // Ingest video
    if (!downloaded) {
      std::string video_path = scanner::download_temp(
          "https://storage.googleapis.com/scanner-data/test/short_video.mp4");
      scanner::Result result;
      std::vector<scanner::FailedVideo> failed_videos;
      result = db_->ingest_videos({"test"}, {video_path}, failed_videos);
      assert(result.success());
      assert(failed_videos.empty());
      downloaded = true;
    }

    // Initialize master and one worker
    scanner::MachineParameters machine_params =
        scanner::default_machine_params();
    db_->start_master(machine_params, master_port);
    db_->start_worker(machine_params, worker_port);

    // Construct job parameters
    params_.memory_pool_config.mutable_cpu()->set_use_pool(false);
    params_.memory_pool_config.mutable_gpu()->set_use_pool(false);
    params_.pipeline_instances_per_node = 1;
    params_.work_item_size = 25;
  }

  void TearDown() { delete db_; }

  scanner::Op* blur_op(scanner::Op* input, scanner::DeviceType device_type) {
    scanner::proto::BlurArgs blur_args;
    blur_args.set_sigma(0.5);
    blur_args.set_kernel_size(3);

    size_t blur_args_size = blur_args.ByteSize();
    char* blur_args_buff = new char[blur_args_size];
    blur_args.SerializeToArray(blur_args_buff, blur_args_size);

    return new scanner::Op("Blur",
                           {scanner::OpInput(input, {"frame"})},
                           device_type, blur_args_buff, blur_args_size);
  }

  void gen_random(char* s, const int len) {
    static const char alphanum[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";

    for (int i = 0; i < len; ++i) {
      s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
    }

    s[len] = 0;
  }

  void run_task(scanner::Task task, scanner::Op* op) {
    char job_name[12];
    gen_random(job_name, 12);
    params_.job_name = job_name;
    params_.task_set.tasks.clear();
    params_.task_set.tasks.push_back(task);
    params_.task_set.output_op = op;

    scanner::Result result = db_->new_job(params_);
    ASSERT_TRUE(result.success()) << "Run job failed: " << result.msg();
  }

  scanner::Task range_task(std::string output_table_name) {
    scanner::Task task;
    task.output_table_name = output_table_name;
    scanner::TableSample sample;
    sample.table_name = "test";
    sample.column_names = {"index", "frame"};
    sample.sampling_function = "Gather";
    scanner::proto::GatherSamplerArgs args;
    auto& gather_sample = *args.add_samples();
    for (int i = 0; i < 100; i += 1) {
      gather_sample.add_rows(i);
    }
    std::vector<scanner::u8> args_data(args.ByteSize());
    args.SerializeToArray(args_data.data(), args_data.size());
    sample.sampling_args = args_data;

    task.samples.push_back(sample);
    return task;
  }

  scanner::JobParameters params_;
  std::unique_ptr<storehouse::StorageConfig> sc_;
  scanner::Database* db_;
};

TEST_F(ScannerTest, NonLinearDAG) {
  scanner::Op* input = scanner::make_input_op({"index", "frame"});

  scanner::Op* hist = new scanner::Op(
      "Histogram",
      {scanner::OpInput(blur_op(input, DeviceType::CPU), {"frame"})},
      scanner::DeviceType::CPU);

  scanner::Op* output =
      scanner::make_output_op({scanner::OpInput(input, {"index"}),
                               scanner::OpInput(hist, {"histogram"})});

  run_task(range_task("NonLinearDAG"), output);
}

#ifdef HAVE_CUDA

TEST_F(ScannerTest, CPUToGPU) {
  scanner::Op* input = scanner::make_input_op({"index", "frame"});

  scanner::Op* hist = new scanner::Op(
      "Histogram",
      {scanner::OpInput(blur_op(input, DeviceType::CPU), {"frame"})},
      scanner::DeviceType::GPU);

  scanner::Op* output =
      scanner::make_output_op({scanner::OpInput(input, {"index"}),
                               scanner::OpInput(hist, {"histogram"})});

  run_task(range_task("CPUToGPU"), output);
}

// TODO: need a GPU blur op
// TEST_F(ScannerTest, GPUToCPU) {
//   scanner::Op *input =
//     scanner::make_input_op({"frame", "frame_info"});

//   scanner::Op *hist = new scanner::Op(
//     "Histogram",
//     {scanner::OpInput(blur_op(input, DeviceType::GPU), {"frame"}),
//      scanner::OpInput(input, {"frame_info"})},
//     scanner::DeviceType::CPU);

//   scanner::Op *output = scanner::make_output_op(
//     {scanner::OpInput(hist, {"histogram"})});

//   run_task(range_task("GPUToCPU"), output);
// }

#endif
}
